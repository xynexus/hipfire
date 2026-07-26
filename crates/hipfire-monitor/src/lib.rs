//! Reusable terminal system monitor for hipfire-adjacent operator surfaces.
//!
//! This crate deliberately owns only terminal rendering and polling state. The
//! telemetry source is `hipfire-sysinfo`, which reads `/proc` and sysfs directly
//! without initializing HIP/ROCm, so the same monitor can run as a standalone
//! binary or be embedded inside `hipfire-tui`.

use std::{
    collections::{BTreeMap, VecDeque},
    env,
    ffi::CString,
    fs, io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    os::unix::ffi::OsStrExt,
    panic,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use hipfire_admin_types::{fmt_bytes, AdminStats, GpuTelemetry, MemPool};
use hipfire_sysinfo::{AmdgpuRegDevice, AmdgpuRegLib, ChipClass, GRBM2_OFFSET, GRBM_OFFSET};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};

const BG: Color = Color::Rgb(17, 20, 23);
const PANEL: Color = Color::Rgb(23, 27, 32);
const PANEL_2: Color = Color::Rgb(41, 49, 58);
const TEXT: Color = Color::Rgb(231, 236, 239);
const MUTED: Color = Color::Rgb(154, 165, 175);
const ACCENT: Color = Color::Rgb(45, 212, 191);
const GREEN: Color = Color::Rgb(102, 217, 139);
const YELLOW: Color = Color::Rgb(238, 190, 95);
const RED: Color = Color::Rgb(255, 95, 104);

const DEFAULT_REFRESH: Duration = Duration::from_millis(800);
const METRIC_LABEL_WIDTH: usize = 10;
const MIN_SPARKLINE_WIDTH: usize = 8;
const NPU_COLUMN_COUNT: usize = 8;
const PEAK_HOLD_REFRESHES: u8 = 3;
const SPARK_CHARS: [char; 8] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇'];

const AMD_DF_PMU: &str = "/sys/bus/event_source/devices/amd_df";
const DF_DRAM_BEAT_BYTES: f64 = 32.0;

#[derive(Debug)]
pub struct MonitorState {
    pub hostname: String,
    pub snapshot: AdminStats,
    pub dram_bandwidth: Option<DramBandwidth>,
    pub dram_bandwidth_meter: Option<RawMeter>,
    pub swap_thrash: SwapThrash,
    pub cpu_cores: Vec<Meter>,
    pub gpu_blocks: Vec<GpuBlockMeter>,
    pub npu_columns: Vec<Meter>,
    pub npu_power: Option<RawMeter>,
    pub network: NetworkTelemetry,
    pub filesystems: Vec<FilesystemUsage>,
    pub view: MonitorView,
    pub metric_view: MetricView,
    pub color_support: TerminalColorSupport,
    uncore: Option<AmdUncoreSampler>,
    swap: SwapSampler,
    cpu: CpuSampler,
    npu: NpuSampler,
    network_sampler: NetworkSampler,
    gpu: Option<GpuBlockSampler>,
    pub last_refresh: Instant,
    pub refresh_interval: Duration,
    pub tick: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct DramBandwidth {
    pub read_mib_s: f64,
    pub write_mib_s: f64,
}

impl DramBandwidth {
    pub fn total_mib_s(self) -> f64 {
        self.read_mib_s + self.write_mib_s
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SwapThrash {
    pub active: bool,
    pub pages_per_sec: f64,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MonitorView {
    Overview,
    Compute,
}

impl MonitorView {
    fn next(self) -> Self {
        match self {
            Self::Overview => Self::Compute,
            Self::Compute => Self::Overview,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Overview => "overview",
            Self::Compute => "compute",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NetworkTelemetry {
    pub receive: Option<RawMeter>,
    pub transmit: Option<RawMeter>,
    pub interfaces: Vec<NetworkInterface>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkInterface {
    pub name: String,
    pub receive_bytes_per_sec: f64,
    pub transmit_bytes_per_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemUsage {
    pub mount_point: PathBuf,
    pub source: String,
    pub fs_type: String,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub total_bytes: u64,
}

impl FilesystemUsage {
    fn percent(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.used_bytes as f64 / self.total_bytes as f64 * 100.0
        }
    }
}

impl MonitorState {
    pub fn new() -> Self {
        Self::with_interval(DEFAULT_REFRESH)
    }

    pub fn with_interval(refresh_interval: Duration) -> Self {
        let mut state = Self {
            hostname: read_hostname(),
            snapshot: AdminStats::default(),
            dram_bandwidth: None,
            dram_bandwidth_meter: None,
            swap_thrash: SwapThrash::default(),
            cpu_cores: Vec::new(),
            gpu_blocks: gpu_block_meters(),
            npu_columns: Vec::new(),
            npu_power: None,
            network: NetworkTelemetry::default(),
            filesystems: Vec::new(),
            view: MonitorView::Overview,
            metric_view: MetricView::Trend,
            color_support: detect_terminal_color_support(),
            uncore: AmdUncoreSampler::open(),
            swap: SwapSampler::new(),
            cpu: CpuSampler::new(),
            npu: NpuSampler::new(),
            network_sampler: NetworkSampler::new(),
            gpu: GpuBlockSampler::open(),
            last_refresh: Instant::now() - refresh_interval,
            refresh_interval,
            tick: 0,
        };
        state.refresh();
        state
    }

    pub fn refresh(&mut self) {
        let elapsed = self.last_refresh.elapsed();
        self.snapshot = hipfire_sysinfo::snapshot(unix_now());
        self.dram_bandwidth = self
            .uncore
            .as_mut()
            .and_then(|sampler| sampler.sample(elapsed));
        if let Some(dram) = self.dram_bandwidth {
            let now = Instant::now();
            let meter = self
                .dram_bandwidth_meter
                .get_or_insert_with(|| RawMeter::new("dram"));
            meter.update(dram.total_mib_s(), now, elapsed);
        }
        self.swap_thrash = self.swap.sample(elapsed);
        self.cpu_cores = self.cpu.sample(elapsed);
        let npu_metrics = self.npu.sample(&self.snapshot, elapsed);
        self.npu_columns = npu_metrics.columns;
        self.npu_power = npu_metrics.power;
        self.network = self.network_sampler.sample(elapsed);
        self.filesystems = read_filesystems();
        if let Some(sampler) = &mut self.gpu {
            self.gpu_blocks = sampler.sample(elapsed);
        }
        self.last_refresh = Instant::now();
    }

    pub fn maybe_refresh(&mut self) {
        if self.last_refresh.elapsed() >= self.refresh_interval {
            self.refresh();
        }
    }

    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        if let Some(sampler) = &mut self.gpu {
            sampler.sample_tick();
        }
        self.maybe_refresh();
    }
}

impl Default for MonitorState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MetricView {
    Trend,
    Text,
}

impl MetricView {
    fn toggled(self) -> Self {
        match self {
            Self::Trend => Self::Text,
            Self::Text => Self::Trend,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Trend => "trend",
            Self::Text => "text",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TerminalColorMode {
    Disabled,
    TrueColor,
    Ansi256,
    Ansi16,
    Monochrome,
    Unknown,
}

impl TerminalColorMode {
    fn label(self) -> &'static str {
        match self {
            Self::Disabled => "off",
            Self::TrueColor => "truecolor",
            Self::Ansi256 => "256",
            Self::Ansi16 => "16",
            Self::Monochrome => "mono",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TerminalColorSupport {
    pub mode: TerminalColorMode,
    pub colors: Option<u32>,
}

impl TerminalColorSupport {
    fn label(self) -> String {
        match self.colors {
            Some(colors)
                if !matches!(
                    self.mode,
                    TerminalColorMode::TrueColor | TerminalColorMode::Disabled
                ) =>
            {
                format!("{} ({colors})", self.mode.label())
            }
            _ => self.mode.label().to_string(),
        }
    }
}

pub fn run_standalone() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal);
    restore_terminal(&mut terminal)?;
    result
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut monitor = MonitorState::new();

    loop {
        terminal.draw(|frame| draw(frame, &monitor, frame.area()))?;
        monitor.tick();

        if event::poll(Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) if handle_key(&mut monitor, key) => break,
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    Ok(())
}

fn handle_key(monitor: &mut MonitorState, key: KeyEvent) -> bool {
    if key.kind == KeyEventKind::Release {
        return false;
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => true,
        KeyCode::Tab => {
            monitor.view = monitor.view.next();
            false
        }
        KeyCode::Char('1') => {
            monitor.view = MonitorView::Overview;
            false
        }
        KeyCode::Char('2') => {
            monitor.view = MonitorView::Compute;
            false
        }
        KeyCode::Char('r') => {
            monitor.refresh();
            false
        }
        KeyCode::Char('z') => {
            monitor.metric_view = monitor.metric_view.toggled();
            false
        }
        _ => false,
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn detect_terminal_color_support() -> TerminalColorSupport {
    let no_color = env::var("NO_COLOR").ok();
    let colorterm = env::var("COLORTERM").ok();
    let term = env::var("TERM").ok();
    let colors = tput_colors();
    TerminalColorSupport {
        mode: classify_terminal_color(
            no_color.as_deref(),
            colorterm.as_deref(),
            term.as_deref(),
            colors,
        ),
        colors,
    }
}

fn tput_colors() -> Option<u32> {
    let output = Command::new("tput").arg("colors").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    text.trim()
        .parse::<i32>()
        .ok()
        .and_then(|v| if v >= 0 { Some(v as u32) } else { None })
}

fn classify_terminal_color(
    no_color: Option<&str>,
    colorterm: Option<&str>,
    term: Option<&str>,
    colors: Option<u32>,
) -> TerminalColorMode {
    if no_color.is_some_and(|v| !v.is_empty()) {
        return TerminalColorMode::Disabled;
    }

    let colorterm = colorterm.unwrap_or_default().to_ascii_lowercase();
    let term = term.unwrap_or_default().to_ascii_lowercase();
    if colorterm.contains("truecolor")
        || colorterm.contains("24bit")
        || term.contains("truecolor")
        || term.contains("24bit")
        || term.contains("direct")
    {
        return TerminalColorMode::TrueColor;
    }

    if colors.is_some_and(|count| count >= 256) || term.contains("256color") {
        return TerminalColorMode::Ansi256;
    }
    if colors.is_some_and(|count| count >= 8) {
        return TerminalColorMode::Ansi16;
    }
    if term == "dumb" || colors == Some(0) {
        return TerminalColorMode::Monochrome;
    }
    TerminalColorMode::Unknown
}

pub fn draw(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    frame.render_widget(Clear, area);
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, monitor, root[0]);
    draw_body(frame, monitor, root[1]);
    draw_footer(frame, monitor, root[2]);
}

fn draw_header(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let gpus = monitor.snapshot.gpus.len();
    let npus = monitor.snapshot.npus.len();
    let host = monitor
        .snapshot
        .host
        .as_ref()
        .map(|h| format!("{:.0}% RAM", h.percent()))
        .unwrap_or_else(|| "RAM unavailable".to_string());
    let title = Line::from(vec![
        Span::styled(
            format!("{}  ", monitor.hostname),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(host, Style::default().fg(MUTED)),
        Span::styled("  •  ", Style::default().fg(PANEL_2)),
        Span::styled(
            format!("{gpus} GPU  {npus} NPU"),
            Style::default().fg(MUTED),
        ),
        Span::raw("    "),
        nav_span("1 overview", monitor.view == MonitorView::Overview),
        Span::raw("  "),
        nav_span("2 compute", monitor.view == MonitorView::Compute),
    ]);
    let header_block = Block::default()
        .title(Span::styled(
            " hipfire monitor ",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(PANEL_2))
        .style(Style::default().bg(PANEL));
    frame.render_widget(
        Paragraph::new(title)
            .block(header_block)
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn nav_span(label: &str, selected: bool) -> Span<'static> {
    let style = if selected {
        Style::default()
            .fg(BG)
            .bg(ACCENT)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(MUTED)
    };
    Span::styled(format!(" {label} "), style)
}

fn draw_footer(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let age = monitor.last_refresh.elapsed().as_secs_f32();
    let help = format!(
        " tab switch  z {}  r refresh  q quit   •   {} view  •   color {}  •   sample {age:.1}s ago",
        monitor.metric_view.label(),
        monitor.view.label(),
        monitor.color_support.label()
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(help, Style::default().fg(MUTED))))
            .style(Style::default().bg(BG)),
        area,
    );
}

fn draw_body(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    match monitor.view {
        MonitorView::Overview => draw_overview(frame, monitor, area),
        MonitorView::Compute => draw_compute(frame, monitor, area),
    }
}

#[derive(Debug, Clone, Copy)]
struct OverviewAreas {
    host: Rect,
    network: Rect,
    gpus: Rect,
    npus: Rect,
    clients: Rect,
}

fn overview_areas(area: Rect) -> OverviewAreas {
    let body = pad(area, 1, 0);
    if body.width >= 96 && body.height >= 24 {
        let rows = Layout::vertical([Constraint::Length(12), Constraint::Min(0)]).split(body);
        let top = Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)])
            .split(rows[0]);
        let lower = Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(rows[1]);
        let accelerators =
            Layout::vertical([Constraint::Percentage(42), Constraint::Percentage(58)])
                .split(lower[0]);
        OverviewAreas {
            host: top[0],
            network: top[1],
            gpus: accelerators[0],
            npus: accelerators[1],
            clients: lower[1],
        }
    } else {
        let rows = Layout::vertical([
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Min(0),
        ])
        .split(body);
        let accelerators =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(rows[2]);
        OverviewAreas {
            host: rows[0],
            network: rows[1],
            gpus: accelerators[0],
            npus: accelerators[1],
            clients: Rect::default(),
        }
    }
}

fn draw_overview(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let areas = overview_areas(area);
    draw_host(frame, monitor, areas.host);
    draw_network(frame, monitor, areas.network);
    draw_gpus(frame, &monitor.snapshot, areas.gpus);
    draw_npu_summary(frame, monitor, areas.npus);
    if areas.clients.width > 0 && areas.clients.height > 0 {
        draw_clients(frame, &monitor.snapshot, areas.clients);
    }
}

#[derive(Debug, Clone, Copy)]
struct ComputeAreas {
    cpu: Rect,
    gpu_blocks: Rect,
    gpus: Rect,
    npus: Rect,
}

fn compute_areas(area: Rect) -> ComputeAreas {
    let body = pad(area, 1, 0);
    if body.width >= 110 {
        let cols = Layout::horizontal([
            Constraint::Percentage(28),
            Constraint::Percentage(40),
            Constraint::Percentage(32),
        ])
        .split(body);
        let accelerators =
            Layout::vertical([Constraint::Percentage(48), Constraint::Percentage(52)])
                .split(cols[2]);
        ComputeAreas {
            cpu: cols[0],
            gpu_blocks: cols[1],
            gpus: accelerators[0],
            npus: accelerators[1],
        }
    } else {
        let rows = Layout::vertical([Constraint::Length(14), Constraint::Min(0)]).split(body);
        let accelerators =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(rows[0]);
        let counters = Layout::horizontal([Constraint::Percentage(36), Constraint::Percentage(64)])
            .split(rows[1]);
        ComputeAreas {
            cpu: counters[0],
            gpu_blocks: counters[1],
            gpus: accelerators[0],
            npus: accelerators[1],
        }
    }
}

fn draw_compute(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let areas = compute_areas(area);
    draw_cpu_cores(frame, monitor, areas.cpu);
    draw_gpu_blocks(frame, monitor, areas.gpu_blocks);
    draw_gpus(frame, &monitor.snapshot, areas.gpus);
    draw_npus(frame, monitor, areas.npus);
}

fn draw_host(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let lines = if let Some(host) = &monitor.snapshot.host {
        let mut lines = vec![
            capacity_line(
                "RAM",
                host.percent(),
                host.used_bytes(),
                host.total_bytes,
                area.width.saturating_sub(2) as usize,
            ),
            Line::from(format!(
                "available {}  •  used {}",
                fmt_bytes(host.available_bytes),
                fmt_bytes(host.used_bytes())
            )),
        ];
        if let Some(dram) = monitor.dram_bandwidth {
            lines.push(Line::from(vec![
                Span::styled("DRAM I/O    ", Style::default().fg(MUTED)),
                Span::styled(
                    format!(
                        "↓ {}  ↑ {}  total {}",
                        format_rate(dram.read_mib_s * 1024.0 * 1024.0),
                        format_rate(dram.write_mib_s * 1024.0 * 1024.0),
                        format_rate(dram.total_mib_s() * 1024.0 * 1024.0)
                    ),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        if monitor.swap_thrash.active {
            lines.push(Line::from(vec![
                Span::styled(
                    "SWAP THRASH ",
                    Style::default().fg(RED).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        "{:.1} pages/s  used {} / {}",
                        monitor.swap_thrash.pages_per_sec,
                        fmt_bytes(monitor.swap_thrash.used_bytes),
                        fmt_bytes(monitor.swap_thrash.total_bytes)
                    ),
                    Style::default().fg(RED),
                ),
            ]));
        } else if monitor.swap_thrash.total_bytes > 0 {
            lines.push(Line::from(vec![
                Span::styled("swap        ", Style::default().fg(MUTED)),
                Span::styled(
                    format!(
                        "{} / {}",
                        fmt_bytes(monitor.swap_thrash.used_bytes),
                        fmt_bytes(monitor.swap_thrash.total_bytes)
                    ),
                    Style::default().fg(MUTED),
                ),
            ]));
        }
        lines.push(Line::from(Span::styled(
            "Storage",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        let available_rows = area
            .height
            .saturating_sub(2)
            .saturating_sub(lines.len() as u16);
        if monitor.filesystems.is_empty() {
            lines.push(Line::from(Span::styled(
                "no local filesystems discovered",
                Style::default().fg(YELLOW),
            )));
        } else {
            lines.extend(
                monitor
                    .filesystems
                    .iter()
                    .take(available_rows as usize)
                    .map(|filesystem| {
                        capacity_line(
                            &mount_label(filesystem),
                            filesystem.percent(),
                            filesystem.used_bytes,
                            filesystem.total_bytes,
                            area.width.saturating_sub(2) as usize,
                        )
                    }),
            );
        }
        lines
    } else {
        vec![Line::from(Span::styled(
            "/proc/meminfo unavailable",
            Style::default().fg(YELLOW),
        ))]
    };
    frame.render_widget(card("Memory & storage", lines), area);
}

fn draw_network(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let width = area.width.saturating_sub(2) as usize;
    let mut lines = Vec::new();
    match (&monitor.network.receive, &monitor.network.transmit) {
        (Some(receive), Some(transmit)) => {
            lines.push(Line::from(vec![
                Span::styled("↓ ", Style::default().fg(ACCENT)),
                Span::styled(
                    format_rate(receive.current),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" receive    ", Style::default().fg(MUTED)),
                Span::styled("↑ ", Style::default().fg(YELLOW)),
                Span::styled(
                    format_rate(transmit.current),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" transmit", Style::default().fg(MUTED)),
            ]));
            lines.push(network_meter_line(
                "rx",
                receive,
                width,
                monitor.metric_view,
            ));
            lines.push(network_meter_line(
                "tx",
                transmit,
                width,
                monitor.metric_view,
            ));
        }
        _ => lines.push(Line::from(Span::styled(
            "waiting for second /proc/net/dev sample",
            Style::default().fg(MUTED),
        ))),
    }

    let interface_rows = area
        .height
        .saturating_sub(2)
        .saturating_sub(lines.len() as u16);
    lines.extend(
        monitor
            .network
            .interfaces
            .iter()
            .take(interface_rows as usize)
            .map(|interface| {
                Line::from(vec![
                    Span::styled(
                        format!("{:<10}", truncate(&interface.name, 10)),
                        Style::default().fg(MUTED),
                    ),
                    Span::styled("↓ ", Style::default().fg(ACCENT)),
                    Span::styled(
                        format!("{:>10}", format_rate(interface.receive_bytes_per_sec)),
                        Style::default().fg(TEXT),
                    ),
                    Span::styled("  ↑ ", Style::default().fg(YELLOW)),
                    Span::styled(
                        format!("{:>10}", format_rate(interface.transmit_bytes_per_sec)),
                        Style::default().fg(TEXT),
                    ),
                ])
            }),
    );
    frame.render_widget(card("Network", lines), area);
}

fn network_meter_line(
    label: &str,
    meter: &RawMeter,
    width: usize,
    view: MetricView,
) -> Line<'static> {
    if view == MetricView::Text {
        return Line::from(vec![
            Span::styled(format!("{label:<3}"), Style::default().fg(MUTED)),
            Span::styled(
                format!(" cur {:>10}", format_rate(meter.current)),
                Style::default().fg(TEXT),
            ),
            Span::styled(
                format!("  avg {:>10}", format_rate(meter.average_60s)),
                Style::default().fg(MUTED),
            ),
            Span::styled(
                format!("  peak {:>10}", format_rate(meter.peak)),
                Style::default().fg(MUTED),
            ),
        ]);
    }
    let value_width = 12;
    let spark_width = width
        .saturating_sub(label.len())
        .saturating_sub(value_width + 2)
        .max(MIN_SPARKLINE_WIDTH);
    let mut spans = vec![Span::styled(
        format!("{label:<3}"),
        Style::default().fg(MUTED),
    )];
    spans.extend(raw_meter_sparkline_spans(meter, spark_width));
    spans.push(Span::styled(
        format!(" {:>value_width$}", format_rate(meter.current)),
        Style::default().fg(TEXT),
    ));
    Line::from(spans)
}

fn draw_gpus(frame: &mut Frame, stats: &AdminStats, area: Rect) {
    let mut lines = Vec::new();
    if stats.gpus.is_empty() {
        lines.push(Line::from(Span::styled(
            "No AMD GPUs visible under /sys/class/drm.",
            Style::default().fg(YELLOW),
        )));
    }
    for gpu in &stats.gpus {
        lines.extend(gpu_lines(gpu));
        lines.push(Line::from(""));
    }
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block("GPU memory and sensors"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn gpu_lines(gpu: &GpuTelemetry) -> Vec<Line<'static>> {
    let class = if gpu.is_integrated() {
        "APU/UMA"
    } else {
        "dGPU"
    };
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", gpu.card),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(class, Style::default().fg(MUTED)),
        Span::raw("  "),
        Span::styled(
            gpu.busy_percent
                .map(|v| format!("busy {v}%"))
                .unwrap_or_else(|| "busy n/a".to_string()),
            Style::default().fg(TEXT),
        ),
    ])];

    if let Some(pool) = gpu.primary_pool() {
        lines.push(pool_line(&pool));
    }
    let secondary = if gpu.is_integrated() {
        gpu.vram_pool()
    } else {
        gpu.gtt_pool()
    };
    if let Some(pool) = secondary {
        lines.push(pool_line(&pool));
    }

    let mut sensors = Vec::new();
    if let Some(temp) = gpu.temp_c {
        sensors.push(format!("{temp:.0} C"));
    }
    if let Some(power) = gpu.power_w {
        sensors.push(format!("{power:.1} W"));
    }
    if let Some(sclk) = gpu.sclk_mhz {
        sensors.push(format!("{sclk} MHz"));
    }
    if let Some(metrics) = &gpu.metrics {
        if let Some(socket) = metrics.socket_power_w {
            sensors.push(format!("socket {socket:.1} W"));
        }
        if let (Some(read), Some(write)) = (metrics.dram_read_mbps, metrics.dram_write_mbps) {
            sensors.push(format!("DRAM r/w {read:.0}/{write:.0} MB/s"));
        }
        if metrics.throttling() {
            sensors.push("THROTTLING".to_string());
        }
    }
    if !sensors.is_empty() {
        lines.push(Line::from(Span::styled(
            sensors.join("  "),
            Style::default().fg(MUTED),
        )));
    }

    lines
}

fn draw_npus(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let mut lines = Vec::new();
    if monitor.snapshot.npus.is_empty() {
        lines.push(Line::from(Span::styled(
            "No AMD XDNA NPU visible under /dev/accel.",
            Style::default().fg(YELLOW),
        )));
    } else {
        for npu in &monitor.snapshot.npus {
            let temp = npu
                .temp_c
                .map(|t| format!("temp {:>3.0} C", t))
                .unwrap_or_else(|| "temp   -".to_string());
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{} ", npu.node),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(temp, Style::default().fg(MUTED)),
                Span::raw("  "),
                Span::styled(
                    format!(
                        "TOPS {}/{}  tasks {}/{}  clk {} MHz",
                        npu.tops_current,
                        npu.tops_max,
                        npu.tasks_current,
                        npu.tasks_max,
                        npu.mp_npu_mhz
                    ),
                    Style::default().fg(MUTED),
                ),
            ]));
        }

        let line_width = area.width.saturating_sub(2) as usize;
        lines.push(metric_header(
            line_width,
            METRIC_LABEL_WIDTH,
            monitor.metric_view,
        ));
        if let Some(power) = &monitor.npu_power {
            lines.push(raw_metric_line(
                "pwr",
                power,
                line_width,
                METRIC_LABEL_WIDTH,
                "W",
                monitor.metric_view,
            ));
        } else {
            lines.push(unavailable_metric_line(
                "pwr",
                line_width,
                METRIC_LABEL_WIDTH,
            ));
        }
        lines.extend((0..NPU_COLUMN_COUNT).map(|idx| {
            let label = format!("{idx:02}");
            if let Some(meter) = monitor.npu_columns.get(idx) {
                metric_line(
                    &label,
                    meter,
                    line_width,
                    METRIC_LABEL_WIDTH,
                    monitor.metric_view,
                )
            } else {
                unavailable_metric_line(&label, line_width, METRIC_LABEL_WIDTH)
            }
        }));
    }
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block("NPU"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn draw_npu_summary(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let mut lines = Vec::new();
    if monitor.snapshot.npus.is_empty() {
        lines.push(Line::from(Span::styled(
            "No AMD XDNA NPU visible under /dev/accel.",
            Style::default().fg(YELLOW),
        )));
    } else {
        for npu in &monitor.snapshot.npus {
            let utilization = format!("{:.0}% util", npu.mean_util_pct);
            let power = npu
                .power_w
                .map(|value| format!("{value:.1} W"))
                .unwrap_or_else(|| "power unavailable".to_string());
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}  ", npu.node),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(utilization, Style::default().fg(TEXT)),
                Span::styled("  •  ", Style::default().fg(PANEL_2)),
                Span::styled(power, Style::default().fg(TEXT)),
            ]));
            lines.push(Line::from(Span::styled(
                format!(
                    "{} / {} TOPS  •  {} / {} tasks  •  {} MHz",
                    npu.tops_current,
                    npu.tops_max,
                    npu.tasks_current,
                    npu.tasks_max,
                    npu.mp_npu_mhz
                ),
                Style::default().fg(MUTED),
            )));
        }
    }
    frame.render_widget(card("NPU summary", lines), area);
}

fn draw_gpu_blocks(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let line_width = area.width.saturating_sub(2) as usize;
    let mut lines = vec![metric_header(
        line_width,
        METRIC_LABEL_WIDTH,
        monitor.metric_view,
    )];
    lines.extend(monitor.gpu_blocks.iter().map(|meter| {
        if let Some(metric) = &meter.metric {
            metric_line(
                &meter.name,
                metric,
                line_width,
                METRIC_LABEL_WIDTH,
                monitor.metric_view,
            )
        } else {
            unavailable_metric_line(&meter.name, line_width, METRIC_LABEL_WIDTH)
        }
    }));
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block("GPU counter blocks"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn draw_cpu_cores(frame: &mut Frame, monitor: &MonitorState, area: Rect) {
    let lines = if monitor.cpu_cores.is_empty() {
        vec![Line::from(Span::styled(
            "waiting for second /proc/stat sample",
            Style::default().fg(MUTED),
        ))]
    } else {
        let line_width = area.width.saturating_sub(2) as usize;
        monitor
            .cpu_cores
            .iter()
            .map(|meter| cpu_metric_line(meter, line_width, monitor.metric_view))
            .collect()
    };
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block(cpu_panel_title(monitor.metric_view)))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL)),
        area,
    );
}

fn cpu_panel_title(view: MetricView) -> &'static str {
    match view {
        MetricView::Text => "CPU cur  avg peak",
        MetricView::Trend => "CPU Utilization",
    }
}

fn cpu_metric_line(meter: &Meter, line_width: usize, view: MetricView) -> Line<'static> {
    match view {
        MetricView::Text => {
            let mut spans = vec![
                Span::styled(format!("{:<2}", meter.name), Style::default().fg(MUTED)),
                Span::raw(" "),
                colored_cpu_value(meter.current),
                Span::raw(" "),
                colored_cpu_value(meter.average_60s),
                Span::raw(" "),
                colored_cpu_value(meter.peak),
            ];
            let used = 2 + 1 + 4 + 1 + 4 + 1 + 4;
            if line_width > used {
                spans.insert(0, Span::raw(" ".repeat(line_width - used)));
            }
            Line::from(spans)
        }
        MetricView::Trend => {
            let sparkline_width = line_width.saturating_sub(3).max(MIN_SPARKLINE_WIDTH);
            let mut spans = vec![
                Span::styled(format!("{:<2}", meter.name), Style::default().fg(MUTED)),
                Span::raw(" "),
            ];
            spans.extend(meter_sparkline_spans(meter, sparkline_width));
            Line::from(spans)
        }
    }
}

fn colored_cpu_value(value: f64) -> Span<'static> {
    Span::styled(
        format!("{:>3.0}%", value.clamp(0.0, 100.0)),
        Style::default().fg(heat_color(value)),
    )
}

fn draw_clients(frame: &mut Frame, stats: &AdminStats, area: Rect) {
    let rows = stats.clients.iter().take(12).map(|client| {
        Row::new([
            client.pid.to_string(),
            client.comm.clone(),
            client.card.clone().unwrap_or_else(|| "?".to_string()),
            fmt_bytes(client.vram_bytes),
            fmt_bytes(client.gtt_bytes),
        ])
        .style(Style::default().fg(TEXT))
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(8),
                Constraint::Min(12),
                Constraint::Length(7),
                Constraint::Length(11),
                Constraint::Length(11),
            ],
        )
        .header(
            Row::new(["PID", "Process", "Card", "VRAM", "GTT"]).style(Style::default().fg(MUTED)),
        )
        .block(block("GPU clients"))
        .style(Style::default().bg(PANEL)),
        area,
    );
}

fn metric_line(
    label: &str,
    meter: &Meter,
    line_width: usize,
    label_width: usize,
    view: MetricView,
) -> Line<'static> {
    if view == MetricView::Text {
        return metric_text_line(
            label,
            [
                metric_value_cell(meter.current, "%", meter.current),
                metric_value_cell(meter.average_60s, "%", meter.average_60s),
                metric_value_cell(meter.peak, "%", meter.peak),
            ],
            line_width,
            label_width,
        );
    }

    let sparkline_width = metric_visual_width(line_width, label_width);
    let mut spans = vec![
        Span::styled(format!("{label:<label_width$}"), Style::default().fg(MUTED)),
        Span::raw(" "),
    ];
    spans.extend(meter_sparkline_spans(meter, sparkline_width));
    Line::from(spans)
}

fn raw_metric_line(
    label: &str,
    meter: &RawMeter,
    line_width: usize,
    label_width: usize,
    suffix: &'static str,
    view: MetricView,
) -> Line<'static> {
    if view == MetricView::Text {
        return metric_text_line(
            label,
            [
                metric_value_cell(meter.current, suffix, meter.normalized_current()),
                metric_value_cell(
                    meter.average_60s,
                    suffix,
                    raw_to_pct(meter.average_60s, meter.scale),
                ),
                metric_value_cell(meter.peak, suffix, raw_to_pct(meter.peak, meter.scale)),
            ],
            line_width,
            label_width,
        );
    }

    let sparkline_width = metric_visual_width(line_width, label_width);
    let mut spans = vec![
        Span::styled(format!("{label:<label_width$}"), Style::default().fg(MUTED)),
        Span::raw(" "),
    ];
    spans.extend(raw_meter_sparkline_spans(meter, sparkline_width));
    Line::from(spans)
}

fn metric_header(line_width: usize, label_width: usize, view: MetricView) -> Line<'static> {
    if view == MetricView::Text {
        return Line::from(Span::styled(
            metric_header_text(line_width, label_width, view),
            Style::default().fg(MUTED),
        ));
    }

    let sparkline_width = metric_visual_width(line_width, label_width);
    Line::from(vec![
        Span::styled(format!("{:<label_width$}", ""), Style::default().fg(MUTED)),
        Span::raw(" "),
        Span::styled(
            format!("{:<sparkline_width$}", "trend"),
            Style::default().fg(MUTED),
        ),
    ])
}

fn metric_header_text(width: usize, label_width: usize, view: MetricView) -> String {
    if view == MetricView::Text {
        return pad_to_width(format!("{:<label_width$}    cur    avg   peak", ""), width);
    }
    let sparkline_width = metric_visual_width(width, label_width);
    format!("{:<label_width$} {:<sparkline_width$}", "", "trend")
}

fn metric_text_line(
    label: &str,
    values: [Span<'static>; 3],
    width: usize,
    label_width: usize,
) -> Line<'static> {
    let mut spans = vec![Span::styled(
        format!("{label:<label_width$}"),
        Style::default().fg(MUTED),
    )];
    spans.extend(values);
    pad_spans_to_width(&mut spans, width);
    Line::from(spans)
}

fn pad_to_width(text: String, width: usize) -> String {
    format!("{text:<width$}")
}

fn pad_spans_to_width(spans: &mut Vec<Span<'static>>, width: usize) {
    let used = spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum::<usize>();
    if width > used {
        spans.push(Span::raw(" ".repeat(width - used)));
    }
}

fn unavailable_metric_line(label: &str, line_width: usize, label_width: usize) -> Line<'static> {
    let value_width = line_width.saturating_sub(label_width + 1);
    Line::from(vec![
        Span::styled(format!("{label:<label_width$}"), Style::default().fg(MUTED)),
        Span::raw(" "),
        Span::styled(
            format!("{:<value_width$}", "unavailable"),
            Style::default().fg(Color::DarkGray),
        ),
    ])
}

fn metric_visual_width(line_width: usize, label_width: usize) -> usize {
    line_width
        .saturating_sub(label_width)
        .saturating_sub(1)
        .max(MIN_SPARKLINE_WIDTH)
}

fn metric_value_cell(value: f64, suffix: &'static str, heat_pct: f64) -> Span<'static> {
    Span::styled(
        format!(" {:>5.1}{suffix}", value),
        Style::default().fg(heat_color(heat_pct)),
    )
}

fn pool_line(pool: &MemPool) -> Line<'static> {
    let pct = pool.percent();
    Line::from(vec![
        Span::styled(format!("{:<16}", pool.label), Style::default().fg(MUTED)),
        Span::styled(bar(pct, 18), Style::default().fg(percent_color(pct))),
        Span::styled(format!(" {:>5.1}%  ", pct), Style::default().fg(TEXT)),
        Span::styled(
            format!(
                "{} / {}",
                fmt_bytes(pool.used_bytes),
                fmt_bytes(pool.total_bytes)
            ),
            Style::default().fg(TEXT),
        ),
    ])
}

fn capacity_line(
    label: &str,
    pct: f64,
    used_bytes: u64,
    total_bytes: u64,
    width: usize,
) -> Line<'static> {
    let label_width = 12.min(width.saturating_sub(1)).max(4);
    let figures = format!("{} / {}", fmt_bytes(used_bytes), fmt_bytes(total_bytes));
    let fixed = label_width + figures.chars().count() + 8;
    let bar_width = width.saturating_sub(fixed).clamp(5, 18);
    Line::from(vec![
        Span::styled(
            format!("{:<label_width$}", truncate(label, label_width)),
            Style::default().fg(MUTED),
        ),
        Span::styled(bar(pct, bar_width), Style::default().fg(percent_color(pct))),
        Span::styled(format!(" {:>4.0}%  ", pct), Style::default().fg(TEXT)),
        Span::styled(figures, Style::default().fg(TEXT)),
    ])
}

fn mount_label(filesystem: &FilesystemUsage) -> String {
    if filesystem.mount_point == Path::new("/") {
        "root /".to_string()
    } else {
        filesystem.mount_point.display().to_string()
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }
    let mut value = text.chars().take(width - 1).collect::<String>();
    value.push('…');
    value
}

fn format_rate(bytes_per_sec: f64) -> String {
    let value = bytes_per_sec.max(0.0);
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * KIB;
    const GIB: f64 = 1024.0 * MIB;
    if value >= GIB {
        format!("{:.1} GiB/s", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MiB/s", value / MIB)
    } else if value >= KIB {
        format!("{:.1} KiB/s", value / KIB)
    } else {
        format!("{value:.0} B/s")
    }
}

fn meter_sparkline_spans(meter: &Meter, width: usize) -> Vec<Span<'static>> {
    sparkline_spans(&meter.history_values(), width)
}

fn raw_meter_sparkline_spans(meter: &RawMeter, width: usize) -> Vec<Span<'static>> {
    sparkline_spans(&meter.normalized_history_values(), width)
}

#[cfg(test)]
fn sparkline(values: &[f64], width: usize) -> String {
    sparkline_cells(values, width)
        .into_iter()
        .map(|(ch, _)| ch)
        .collect()
}

fn sparkline_spans(values: &[f64], width: usize) -> Vec<Span<'static>> {
    sparkline_cells(values, width)
        .into_iter()
        .map(|(ch, value)| Span::styled(ch.to_string(), Style::default().fg(heat_color(value))))
        .collect()
}

fn sparkline_cells(values: &[f64], width: usize) -> Vec<(char, f64)> {
    if width == 0 {
        return Vec::new();
    }
    if values.is_empty() {
        return vec![(' ', 0.0); width];
    }
    let samples = values.len().min(width);
    let start = values.len().saturating_sub(samples);
    let mut out = vec![(' ', 0.0); width.saturating_sub(samples)];
    out.extend(
        values[start..]
            .iter()
            .map(|value| (spark_char(*value), *value)),
    );
    out
}

fn spark_char(pct: f64) -> char {
    let idx = ((pct.clamp(0.0, 100.0) / 100.0) * ((SPARK_CHARS.len() - 1) as f64)).round() as usize;
    SPARK_CHARS[idx]
}

fn heat_color(pct: f64) -> Color {
    let t = (pct.clamp(0.0, 100.0) / 100.0).clamp(0.0, 1.0);
    let mix = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * t).round() as u8;
    match (GREEN, RED) {
        (Color::Rgb(gr, gg, gb), Color::Rgb(rr, rg, rb)) => {
            Color::Rgb(mix(gr, rr), mix(gg, rg), mix(gb, rb))
        }
        _ => percent_color(pct),
    }
}

fn percent_color(pct: f64) -> Color {
    if pct >= 90.0 {
        RED
    } else if pct >= 75.0 {
        YELLOW
    } else {
        GREEN
    }
}

fn bar(pct: f64, width: usize) -> String {
    let pct = pct.clamp(0.0, 100.0);
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    format!(
        "{}{}",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

fn card(title: &str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(block(title))
        .style(Style::default().fg(TEXT).bg(PANEL))
        .wrap(Wrap { trim: false })
}

fn block(title: &str) -> Block<'static> {
    Block::default()
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PANEL_2))
        .style(Style::default().bg(PANEL))
}

fn pad(area: Rect, x: u16, y: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(x),
        y: area.y.saturating_add(y),
        width: area.width.saturating_sub(x * 2),
        height: area.height.saturating_sub(y * 2),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Clone)]
pub struct Meter {
    pub name: String,
    pub current: f64,
    pub peak: f64,
    pub average_60s: f64,
    peak_hold: u8,
    history: VecDeque<(Instant, f64)>,
}

impl Meter {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            current: 0.0,
            peak: 0.0,
            average_60s: 0.0,
            peak_hold: 0,
            history: VecDeque::new(),
        }
    }

    fn update(&mut self, value: f64, now: Instant, elapsed: Duration) {
        let _ = elapsed;
        self.current = value.clamp(0.0, 100.0);
        if self.current >= self.peak {
            self.peak = self.current;
            self.peak_hold = PEAK_HOLD_REFRESHES;
        } else if self.peak_hold > 0 {
            self.peak_hold -= 1;
        } else {
            self.peak = self.current;
        }
        self.history.push_back((now, self.current));
        while self
            .history
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(60))
        {
            self.history.pop_front();
        }
        self.average_60s = if self.history.is_empty() {
            self.current
        } else {
            self.history.iter().map(|(_, v)| *v).sum::<f64>() / self.history.len() as f64
        };
    }

    fn history_values(&self) -> Vec<f64> {
        if self.history.is_empty() {
            vec![self.current]
        } else {
            self.history.iter().map(|(_, v)| *v).collect()
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawMeter {
    pub name: String,
    pub current: f64,
    pub peak: f64,
    pub average_60s: f64,
    scale: f64,
    peak_hold: u8,
    history: VecDeque<(Instant, f64)>,
}

impl RawMeter {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            current: 0.0,
            peak: 0.0,
            average_60s: 0.0,
            scale: 1.0,
            peak_hold: 0,
            history: VecDeque::new(),
        }
    }

    fn update(&mut self, value: f64, now: Instant, elapsed: Duration) {
        let _ = elapsed;
        self.current = value.max(0.0);
        if self.current >= self.peak {
            self.peak = self.current;
            self.peak_hold = PEAK_HOLD_REFRESHES;
        } else if self.peak_hold > 0 {
            self.peak_hold -= 1;
        } else {
            self.peak = self.current;
        }
        self.history.push_back((now, self.current));
        while self
            .history
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(60))
        {
            self.history.pop_front();
        }
        self.average_60s = if self.history.is_empty() {
            self.current
        } else {
            self.history.iter().map(|(_, v)| *v).sum::<f64>() / self.history.len() as f64
        };
        self.scale = self
            .history
            .iter()
            .map(|(_, v)| *v)
            .fold(self.peak, f64::max)
            .max(self.average_60s)
            .max(1.0);
    }

    fn normalized_current(&self) -> f64 {
        raw_to_pct(self.current, self.scale)
    }

    fn normalized_history_values(&self) -> Vec<f64> {
        if self.history.is_empty() {
            vec![self.normalized_current()]
        } else {
            self.history
                .iter()
                .map(|(_, v)| raw_to_pct(*v, self.scale))
                .collect()
        }
    }
}

fn raw_to_pct(value: f64, scale: f64) -> f64 {
    if scale <= 0.0 {
        0.0
    } else {
        (value / scale * 100.0).clamp(0.0, 100.0)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NetworkCounters {
    receive_bytes: u64,
    transmit_bytes: u64,
}

#[derive(Debug, Default)]
struct NetworkSampler {
    previous: BTreeMap<String, NetworkCounters>,
    receive: Option<RawMeter>,
    transmit: Option<RawMeter>,
}

impl NetworkSampler {
    fn new() -> Self {
        Self {
            previous: read_network_counters(),
            receive: None,
            transmit: None,
        }
    }

    fn sample(&mut self, elapsed: Duration) -> NetworkTelemetry {
        let current = read_network_counters();
        let interfaces = network_rates(&self.previous, &current, elapsed);
        self.previous = current;
        let receive = interfaces
            .iter()
            .map(|interface| interface.receive_bytes_per_sec)
            .sum::<f64>();
        let transmit = interfaces
            .iter()
            .map(|interface| interface.transmit_bytes_per_sec)
            .sum::<f64>();
        let now = Instant::now();
        self.receive
            .get_or_insert_with(|| RawMeter::new("rx"))
            .update(receive, now, elapsed);
        self.transmit
            .get_or_insert_with(|| RawMeter::new("tx"))
            .update(transmit, now, elapsed);
        NetworkTelemetry {
            receive: self.receive.clone(),
            transmit: self.transmit.clone(),
            interfaces,
        }
    }
}

fn read_network_counters() -> BTreeMap<String, NetworkCounters> {
    fs::read_to_string("/proc/net/dev")
        .ok()
        .map(|text| parse_network_counters(&text))
        .unwrap_or_default()
}

fn parse_network_counters(text: &str) -> BTreeMap<String, NetworkCounters> {
    text.lines()
        .filter_map(|line| {
            let (name, values) = line.split_once(':')?;
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            let values = values
                .split_whitespace()
                .filter_map(|value| value.parse::<u64>().ok())
                .collect::<Vec<_>>();
            Some((
                name.to_string(),
                NetworkCounters {
                    receive_bytes: *values.first()?,
                    transmit_bytes: *values.get(8)?,
                },
            ))
        })
        .collect()
}

fn network_rates(
    previous: &BTreeMap<String, NetworkCounters>,
    current: &BTreeMap<String, NetworkCounters>,
    elapsed: Duration,
) -> Vec<NetworkInterface> {
    let seconds = elapsed.as_secs_f64();
    let mut rates = current
        .iter()
        .filter(|(name, _)| name.as_str() != "lo")
        .map(|(name, counters)| {
            let old = previous.get(name).copied().unwrap_or(*counters);
            NetworkInterface {
                name: name.clone(),
                receive_bytes_per_sec: if seconds > 0.0 {
                    counters.receive_bytes.saturating_sub(old.receive_bytes) as f64 / seconds
                } else {
                    0.0
                },
                transmit_bytes_per_sec: if seconds > 0.0 {
                    counters.transmit_bytes.saturating_sub(old.transmit_bytes) as f64 / seconds
                } else {
                    0.0
                },
            }
        })
        .collect::<Vec<_>>();
    rates.sort_by(|left, right| {
        let left_total = left.receive_bytes_per_sec + left.transmit_bytes_per_sec;
        let right_total = right.receive_bytes_per_sec + right.transmit_bytes_per_sec;
        right_total
            .total_cmp(&left_total)
            .then_with(|| left.name.cmp(&right.name))
    });
    rates
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MountInfo {
    mount_point: PathBuf,
    fs_type: String,
    source: String,
}

fn read_filesystems() -> Vec<FilesystemUsage> {
    let Ok(text) = fs::read_to_string("/proc/self/mountinfo") else {
        return Vec::new();
    };
    let mut filesystems = parse_mountinfo(&text)
        .into_iter()
        .filter(is_operator_filesystem)
        .filter_map(|mount| filesystem_usage(&mount))
        .collect::<Vec<_>>();
    filesystems.sort_by(|left, right| {
        let left_root = left.mount_point == Path::new("/");
        let right_root = right.mount_point == Path::new("/");
        right_root
            .cmp(&left_root)
            .then_with(|| left.mount_point.cmp(&right.mount_point))
    });
    filesystems.dedup_by(|left, right| left.mount_point == right.mount_point);
    filesystems
}

fn parse_mountinfo(text: &str) -> Vec<MountInfo> {
    text.lines()
        .filter_map(|line| {
            let (mount, filesystem) = line.split_once(" - ")?;
            let mount_fields = mount.split_whitespace().collect::<Vec<_>>();
            let filesystem_fields = filesystem.split_whitespace().collect::<Vec<_>>();
            Some(MountInfo {
                mount_point: PathBuf::from(unescape_mount_field(*mount_fields.get(4)?)),
                fs_type: filesystem_fields.first()?.to_string(),
                source: unescape_mount_field(*filesystem_fields.get(1)?),
            })
        })
        .collect()
}

fn unescape_mount_field(value: &str) -> String {
    value
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

fn is_operator_filesystem(mount: &MountInfo) -> bool {
    matches!(
        mount.fs_type.as_str(),
        "ext2"
            | "ext3"
            | "ext4"
            | "xfs"
            | "btrfs"
            | "zfs"
            | "f2fs"
            | "bcachefs"
            | "nfs"
            | "nfs4"
            | "cifs"
    ) || mount.source.starts_with("/dev/")
        || (mount.mount_point == Path::new("/") && mount.fs_type == "overlay")
}

fn filesystem_usage(mount: &MountInfo) -> Option<FilesystemUsage> {
    let path = CString::new(mount.mount_point.as_os_str().as_bytes()).ok()?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a valid NUL-terminated path and `statvfs` initializes
    // the output on a zero return code.
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful `statvfs` initialized the structure above.
    let stats = unsafe { stats.assume_init() };
    let block_size = stats.f_frsize as u64;
    let total_bytes = (stats.f_blocks as u64).saturating_mul(block_size);
    let free_bytes = (stats.f_bfree as u64).saturating_mul(block_size);
    let available_bytes = (stats.f_bavail as u64).saturating_mul(block_size);
    Some(FilesystemUsage {
        mount_point: mount.mount_point.clone(),
        source: mount.source.clone(),
        fs_type: mount.fs_type.clone(),
        used_bytes: total_bytes.saturating_sub(free_bytes),
        available_bytes,
        total_bytes,
    })
}

fn read_hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "local host".to_string())
}

#[derive(Debug, Default)]
struct NpuSampler {
    columns: Vec<Meter>,
    power: Option<RawMeter>,
}

#[derive(Debug, Default)]
struct NpuSample {
    columns: Vec<Meter>,
    power: Option<RawMeter>,
}

impl NpuSampler {
    fn new() -> Self {
        Self {
            columns: (0..NPU_COLUMN_COUNT)
                .map(|idx| Meter::new(format!("{idx:02}")))
                .collect(),
            power: None,
        }
    }

    fn sample(&mut self, stats: &AdminStats, elapsed: Duration) -> NpuSample {
        let now = Instant::now();
        let columns = fold_npu_columns(stats);
        for (idx, meter) in self.columns.iter_mut().enumerate() {
            if let Some(value) = columns.get(idx) {
                meter.update((*value).into(), now, elapsed);
            }
        }

        let power = stats
            .npus
            .iter()
            .filter_map(|npu| npu.power_w)
            .reduce(f64::max);
        if let Some(value) = power {
            self.power
                .get_or_insert_with(|| RawMeter::new("pwr"))
                .update(value, now, elapsed);
        }

        NpuSample {
            columns: if !columns.is_empty() {
                self.columns.clone()
            } else {
                Default::default()
            },
            power: self.power.clone().filter(|_| power.is_some()),
        }
    }
}

fn fold_npu_columns(stats: &AdminStats) -> Vec<u32> {
    let mut values = vec![0; NPU_COLUMN_COUNT];
    let mut any = false;
    for npu in &stats.npus {
        for (idx, value) in npu.columns_pct.iter().take(NPU_COLUMN_COUNT).enumerate() {
            values[idx] = values[idx].max((*value).min(100));
            any = true;
        }
    }
    if any {
        values
    } else {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct GpuBlockMeter {
    pub name: String,
    pub metric: Option<Meter>,
}

fn gpu_block_meters() -> Vec<GpuBlockMeter> {
    GPU_BLOCK_LABELS
        .into_iter()
        .map(|name| GpuBlockMeter {
            name: name.to_string(),
            metric: None,
        })
        .collect()
}

const GPU_BLOCK_LABELS: [&str; 9] = [
    "GFX Pipe",
    "Tex Pipe",
    "SPI",
    "RLC",
    "TCP",
    "UTCL2",
    "EA",
    "CP Fetch",
    "CP Compute",
];

const GRBM_INDEX: &[(&str, usize)] = &[("GFX Pipe", 31), ("Tex Pipe", 14), ("SPI", 22)];

const GRBM2_INDEX: &[(&str, usize)] = &[
    ("RLC", 24),
    ("TCP", 25),
    ("CP Fetch", 28),
    ("CP Compute", 29),
];

const GFX9_GRBM2_INDEX: &[(&str, usize)] = &[
    ("RLC", 24),
    ("TCP", 25),
    ("UTCL2", 15),
    ("EA", 16),
    ("CP Fetch", 28),
    ("CP Compute", 29),
];

const GFX10_GRBM2_INDEX: &[(&str, usize)] = &[
    ("RLC", 24),
    ("TCP", 25),
    ("UTCL2", 15),
    ("EA", 16),
    ("CP Fetch", 28),
    ("CP Compute", 29),
];

const GFX10_3_GRBM2_INDEX: &[(&str, usize)] = &[
    ("RLC", 26),
    ("TCP", 27),
    ("UTCL2", 15),
    ("EA", 16),
    ("CP Fetch", 28),
    ("CP Compute", 29),
];

const GFX12_GRBM2_INDEX: &[(&str, usize)] = &[
    ("RLC", 26),
    ("TCP", 27),
    ("UTCL2", 15),
    ("EA", 16),
    ("CP Fetch", 28),
    ("CP Compute", 29),
];

struct GpuBlockSampler {
    devices: Vec<GpuCounterDevice>,
    meters: BTreeMap<String, Meter>,
    sample_count: u64,
}

impl std::fmt::Debug for GpuBlockSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBlockSampler")
            .field("devices", &self.devices.len())
            .field("meters", &self.meters.keys().collect::<Vec<_>>())
            .field("sample_count", &self.sample_count)
            .finish()
    }
}

impl GpuBlockSampler {
    fn open() -> Option<Self> {
        let libdrm = AmdgpuRegLib::load()?;
        let devices = discover_amdgpu_render_nodes()
            .into_iter()
            .filter_map(|path| libdrm.open_device(&path).map(GpuCounterDevice::new))
            .collect::<Vec<_>>();
        if devices.is_empty() {
            return None;
        }
        let meters = GPU_BLOCK_LABELS
            .into_iter()
            .map(|label| (label.to_string(), Meter::new(label)))
            .collect();
        Some(Self {
            devices,
            meters,
            sample_count: 0,
        })
    }

    fn sample_tick(&mut self) {
        let mut sampled = false;
        for device in &mut self.devices {
            sampled |= device.sample();
        }
        if sampled {
            self.sample_count = self.sample_count.saturating_add(1);
        }
    }

    fn sample(&mut self, elapsed: Duration) -> Vec<GpuBlockMeter> {
        if self.sample_count == 0 {
            return gpu_block_meters();
        }

        let now = Instant::now();
        let values = self.fold_usage();
        for (label, meter) in &mut self.meters {
            if let Some(value) = values.get(label) {
                meter.update(*value, now, elapsed);
            }
        }

        for device in &mut self.devices {
            device.clear();
        }
        self.sample_count = 0;

        GPU_BLOCK_LABELS
            .into_iter()
            .map(|label| GpuBlockMeter {
                name: label.to_string(),
                metric: values
                    .contains_key(label)
                    .then(|| self.meters.get(label).cloned())
                    .flatten(),
            })
            .collect()
    }

    fn fold_usage(&self) -> BTreeMap<String, f64> {
        let mut values = BTreeMap::new();
        for device in &self.devices {
            for (name, value) in device.usage() {
                values
                    .entry(name)
                    .and_modify(|current: &mut f64| *current = current.max(value))
                    .or_insert(value);
            }
        }
        values
    }
}

struct GpuCounterDevice {
    device: AmdgpuRegDevice,
    grbm: RegisterSampler,
    grbm2: RegisterSampler,
}

impl GpuCounterDevice {
    fn new(device: AmdgpuRegDevice) -> Self {
        let chip_class = device.chip_class();
        Self {
            device,
            grbm: RegisterSampler::new(GRBM_OFFSET, GRBM_INDEX),
            grbm2: RegisterSampler::new(GRBM2_OFFSET, grbm2_index(chip_class)),
        }
    }

    fn sample(&mut self) -> bool {
        let grbm = self.grbm.sample(&self.device);
        let grbm2 = self.grbm2.sample(&self.device);
        grbm || grbm2
    }

    fn clear(&mut self) {
        self.grbm.clear();
        self.grbm2.clear();
    }

    fn usage(&self) -> Vec<(String, f64)> {
        self.grbm
            .usage()
            .into_iter()
            .chain(self.grbm2.usage())
            .collect()
    }
}

#[derive(Debug)]
struct RegisterSampler {
    offset: u32,
    indexes: Vec<RegisterIndex>,
    hits: [u64; 32],
    samples: u64,
}

impl RegisterSampler {
    fn new(offset: u32, indexes: &[(&str, usize)]) -> Self {
        Self {
            offset,
            indexes: indexes
                .iter()
                .map(|(name, bit)| RegisterIndex {
                    name: (*name).to_string(),
                    bit: *bit,
                })
                .collect(),
            hits: [0; 32],
            samples: 0,
        }
    }

    fn sample(&mut self, device: &AmdgpuRegDevice) -> bool {
        let Some(value) = device.read_mm_register(self.offset) else {
            return false;
        };
        self.samples = self.samples.saturating_add(1);
        for index in &self.indexes {
            if ((value >> index.bit) & 1) != 0 {
                self.hits[index.bit] = self.hits[index.bit].saturating_add(1);
            }
        }
        true
    }

    fn clear(&mut self) {
        self.hits = [0; 32];
        self.samples = 0;
    }

    fn usage(&self) -> Vec<(String, f64)> {
        if self.samples == 0 {
            return Vec::new();
        }
        self.indexes
            .iter()
            .map(|index| {
                (
                    index.name.clone(),
                    (self.hits[index.bit] as f64 / self.samples as f64) * 100.0,
                )
            })
            .collect()
    }
}

#[derive(Debug)]
struct RegisterIndex {
    name: String,
    bit: usize,
}

fn discover_amdgpu_render_nodes() -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir("/sys/bus/pci/drivers/amdgpu") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let pci_id = entry.file_name();
            let pci_id = pci_id.to_string_lossy();
            if pci_id.len() != "0000:00:00.0".len() {
                return None;
            }
            find_render_node(entry.path().join("drm"))
        })
        .collect()
}

fn find_render_node(drm_path: PathBuf) -> Option<PathBuf> {
    let mut render_nodes = fs::read_dir(drm_path)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("renderD")
                .and_then(|minor| minor.parse::<u32>().ok())
                .map(|minor| (minor, PathBuf::from(format!("/dev/dri/renderD{minor}"))))
        })
        .collect::<Vec<_>>();
    render_nodes.sort_by_key(|(minor, _)| *minor);
    render_nodes.into_iter().map(|(_, path)| path).next()
}

fn grbm2_index(chip_class: ChipClass) -> &'static [(&'static str, usize)] {
    if ChipClass::Gfx12 <= chip_class {
        GFX12_GRBM2_INDEX
    } else if ChipClass::Gfx10_3 <= chip_class {
        GFX10_3_GRBM2_INDEX
    } else if ChipClass::Gfx10 <= chip_class {
        GFX10_GRBM2_INDEX
    } else if ChipClass::Gfx9 <= chip_class {
        GFX9_GRBM2_INDEX
    } else {
        GRBM2_INDEX
    }
}

#[derive(Debug, Default)]
struct SwapSampler {
    prev: Option<VmstatSwap>,
}

impl SwapSampler {
    fn new() -> Self {
        Self {
            prev: read_vmstat_swap(),
        }
    }

    fn sample(&mut self, elapsed: Duration) -> SwapThrash {
        let mem = read_swap_memory().unwrap_or_default();
        let current = read_vmstat_swap();
        let pages_per_sec = match (self.prev, current) {
            (Some(prev), Some(next)) => {
                let delta = next
                    .pswpin
                    .saturating_sub(prev.pswpin)
                    .saturating_add(next.pswpout.saturating_sub(prev.pswpout));
                if elapsed.as_secs_f64() > 0.0 {
                    delta as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        self.prev = current;
        SwapThrash {
            active: mem.used_bytes > 0 && pages_per_sec > 0.0,
            pages_per_sec,
            used_bytes: mem.used_bytes,
            total_bytes: mem.total_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SwapMemory {
    total_bytes: u64,
    used_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct VmstatSwap {
    pswpin: u64,
    pswpout: u64,
}

fn read_swap_memory() -> Option<SwapMemory> {
    parse_swap_memory(&fs::read_to_string("/proc/meminfo").ok()?)
}

fn parse_swap_memory(text: &str) -> Option<SwapMemory> {
    let mut total_kib = None;
    let mut free_kib = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok());
        match key.trim() {
            "SwapTotal" => total_kib = value,
            "SwapFree" => free_kib = value,
            _ => {}
        }
    }
    let total = total_kib?;
    let free = free_kib?.min(total);
    let total_bytes = total * 1024;
    Some(SwapMemory {
        total_bytes,
        used_bytes: total_bytes.saturating_sub(free * 1024),
    })
}

fn read_vmstat_swap() -> Option<VmstatSwap> {
    parse_vmstat_swap(&fs::read_to_string("/proc/vmstat").ok()?)
}

fn parse_vmstat_swap(text: &str) -> Option<VmstatSwap> {
    let mut pswpin = None;
    let mut pswpout = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else {
            continue;
        };
        let value = fields.next().and_then(|v| v.parse::<u64>().ok());
        match key {
            "pswpin" => pswpin = value,
            "pswpout" => pswpout = value,
            _ => {}
        }
    }
    Some(VmstatSwap {
        pswpin: pswpin?,
        pswpout: pswpout?,
    })
}

#[derive(Debug, Default)]
struct CpuSampler {
    prev: Vec<CpuTimes>,
    meters: Vec<Meter>,
}

impl CpuSampler {
    fn new() -> Self {
        Self {
            prev: read_cpu_times(),
            meters: Vec::new(),
        }
    }

    fn sample(&mut self, elapsed: Duration) -> Vec<Meter> {
        let now = Instant::now();
        let current = read_cpu_times();
        if self.prev.is_empty() || current.is_empty() {
            self.prev = current;
            return Vec::new();
        }
        if self.meters.len() != current.len() {
            self.meters = (0..current.len())
                .map(|idx| Meter::new(format!("{idx:02}")))
                .collect();
        }
        for (idx, (prev, next)) in self.prev.iter().zip(current.iter()).enumerate() {
            let pct = cpu_usage_percent(*prev, *next);
            if let Some(meter) = self.meters.get_mut(idx) {
                meter.update(pct, now, elapsed);
            }
        }
        self.prev = current;
        self.meters.clone()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuTimes {
    idle: u64,
    total: u64,
}

fn read_cpu_times() -> Vec<CpuTimes> {
    let Ok(text) = fs::read_to_string("/proc/stat") else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let label = fields.next()?;
            let idx = label.strip_prefix("cpu")?;
            if idx.is_empty() || !idx.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let values = fields
                .take(10)
                .filter_map(|v| v.parse::<u64>().ok())
                .collect::<Vec<_>>();
            if values.len() < 4 {
                return None;
            }
            let idle = values.get(3).copied().unwrap_or(0) + values.get(4).copied().unwrap_or(0);
            let total = values.iter().copied().sum();
            Some(CpuTimes { idle, total })
        })
        .collect()
}

fn cpu_usage_percent(prev: CpuTimes, next: CpuTimes) -> f64 {
    let total = next.total.saturating_sub(prev.total);
    if total == 0 {
        return 0.0;
    }
    let idle = next.idle.saturating_sub(prev.idle);
    ((total.saturating_sub(idle) as f64) / total as f64) * 100.0
}

#[derive(Debug)]
struct AmdUncoreSampler {
    read: Vec<PerfCounter>,
    write: Vec<PerfCounter>,
    prev_read: u64,
    prev_write: u64,
}

impl AmdUncoreSampler {
    fn open() -> Option<Self> {
        let pmu_type = fs::read_to_string(Path::new(AMD_DF_PMU).join("type"))
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()?;
        let read = open_df_dram_counters(pmu_type, 0xffe);
        let write = open_df_dram_counters(pmu_type, 0xfff);
        if read.is_empty() || write.is_empty() {
            return None;
        }
        let mut sampler = Self {
            read,
            write,
            prev_read: 0,
            prev_write: 0,
        };
        sampler.prev_read = sampler.read_count();
        sampler.prev_write = sampler.write_count();
        Some(sampler)
    }

    fn sample(&mut self, elapsed: Duration) -> Option<DramBandwidth> {
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 {
            return None;
        }
        let read = self.read_count();
        let write = self.write_count();
        let read_delta = read.saturating_sub(self.prev_read);
        let write_delta = write.saturating_sub(self.prev_write);
        self.prev_read = read;
        self.prev_write = write;

        let to_mib_s = |beats: u64| beats as f64 * DF_DRAM_BEAT_BYTES / (1024.0 * 1024.0) / secs;
        Some(DramBandwidth {
            read_mib_s: to_mib_s(read_delta),
            write_mib_s: to_mib_s(write_delta),
        })
    }

    fn read_count(&self) -> u64 {
        self.read.iter().filter_map(PerfCounter::read).sum()
    }

    fn write_count(&self) -> u64 {
        self.write.iter().filter_map(PerfCounter::read).sum()
    }
}

fn open_df_dram_counters(pmu_type: u32, umask: u64) -> Vec<PerfCounter> {
    df_dram_events()
        .iter()
        .filter_map(|event| PerfCounter::open(pmu_type, encode_amd_df_config(*event, umask)))
        .collect()
}

fn df_dram_events() -> [u64; 12] {
    [
        0x01f, 0x05f, 0x09f, 0x0df, 0x11f, 0x15f, 0x19f, 0x1df, 0x21f, 0x25f, 0x29f, 0x2df,
    ]
}

fn encode_amd_df_config(event: u64, umask: u64) -> u64 {
    (event & 0xff)
        | (((event >> 8) & 0x3f) << 32)
        | ((umask & 0xff) << 8)
        | (((umask >> 8) & 0x0f) << 24)
}

#[derive(Debug)]
struct PerfCounter {
    fd: OwnedFd,
}

impl PerfCounter {
    fn open(pmu_type: u32, config: u64) -> Option<Self> {
        let mut attr = PerfEventAttr::new(pmu_type, config);
        // SAFETY: perf_event_open is called with a valid pointer to our local
        // perf_event_attr-compatible prefix. The kernel copies the struct.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &mut attr as *mut PerfEventAttr,
                -1i32,
                0i32,
                -1i32,
                0u64,
            )
        };
        if fd < 0 {
            return None;
        }
        // SAFETY: a non-negative perf_event_open return is a newly owned fd.
        Some(Self {
            fd: unsafe { OwnedFd::from_raw_fd(fd as i32) },
        })
    }

    fn read(&self) -> Option<u64> {
        let mut value = 0u64;
        // SAFETY: reads exactly one u64 into a valid pointer from an owned fd.
        let n = unsafe {
            libc::read(
                self.fd.as_raw_fd(),
                &mut value as *mut u64 as *mut libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };
        (n == std::mem::size_of::<u64>() as isize).then_some(value)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period_or_freq: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events_or_watermark: u32,
    bp_type: u32,
    config1: u64,
}

impl PerfEventAttr {
    fn new(type_: u32, config: u64) -> Self {
        Self {
            type_,
            size: std::mem::size_of::<Self>() as u32,
            config,
            sample_period_or_freq: 0,
            sample_type: 0,
            read_format: 0,
            flags: 0,
            wakeup_events_or_watermark: 0,
            bp_type: 0,
            config1: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_clamps_and_preserves_width() {
        assert_eq!(bar(-1.0, 5), "-----");
        assert_eq!(bar(50.0, 6), "###---");
        assert_eq!(bar(101.0, 5), "#####");
    }

    #[test]
    fn sparkline_preserves_width_and_encodes_levels() {
        assert_eq!(sparkline(&[], 4), "    ");
        assert_eq!(sparkline(&[0.0, 50.0, 100.0], 5), "   ▄▇");
        assert_eq!(sparkline(&[0.0, 25.0, 50.0, 75.0, 100.0], 3), "▄▅▇");
        assert_eq!(spark_char(0.0), ' ');
        assert_eq!(spark_char(100.0), '▇');
    }

    #[test]
    fn sparkline_spans_use_green_to_red_heat_colors() {
        let spans = sparkline_spans(&[0.0, 50.0, 100.0], 3);
        assert_eq!(spans[0].style.fg, Some(GREEN));
        assert_eq!(spans[2].style.fg, Some(RED));
        assert_ne!(spans[1].style.fg, Some(GREEN));
        assert_ne!(spans[1].style.fg, Some(RED));
    }

    #[test]
    fn metric_visual_view_uses_sparkline_without_average_marker() {
        let mut meter = Meter::new("x");
        let now = Instant::now();
        meter.update(0.0, now, Duration::from_millis(800));
        meter.update(
            50.0,
            now + Duration::from_millis(800),
            Duration::from_millis(800),
        );
        meter.update(
            100.0,
            now + Duration::from_millis(1600),
            Duration::from_millis(800),
        );
        meter.average_60s = 25.0;
        let trend = meter_sparkline_spans(&meter, 9)
            .into_iter()
            .flat_map(|span| span.content.chars().collect::<Vec<_>>())
            .collect::<String>();
        assert!(trend.contains(' '));
        assert!(trend.contains('▇'));
        assert!(!trend.contains('◆'));
        assert_eq!(trend.chars().count(), 9);
    }

    #[test]
    fn meter_peak_holds_then_snaps_to_current() {
        let now = Instant::now();
        let elapsed = Duration::from_millis(800);
        let mut meter = Meter::new("x");
        meter.update(90.0, now, elapsed);
        assert_eq!(meter.peak, 90.0);
        for idx in 0..PEAK_HOLD_REFRESHES {
            meter.update(10.0, now + elapsed * (idx as u32 + 1), elapsed);
            assert_eq!(meter.peak, 90.0);
        }
        meter.update(
            10.0,
            now + elapsed * (PEAK_HOLD_REFRESHES as u32 + 1),
            elapsed,
        );
        assert_eq!(meter.peak, 10.0);
    }

    #[test]
    fn raw_meter_peak_holds_then_snaps_to_current() {
        let now = Instant::now();
        let elapsed = Duration::from_millis(800);
        let mut meter = RawMeter::new("pwr");
        meter.update(8.0, now, elapsed);
        assert_eq!(meter.peak, 8.0);
        for idx in 0..PEAK_HOLD_REFRESHES {
            meter.update(2.0, now + elapsed * (idx as u32 + 1), elapsed);
            assert_eq!(meter.peak, 8.0);
        }
        meter.update(
            2.0,
            now + elapsed * (PEAK_HOLD_REFRESHES as u32 + 1),
            elapsed,
        );
        assert_eq!(meter.peak, 2.0);
    }

    #[test]
    fn encodes_amd_df_split_event_and_umask_fields() {
        assert_eq!(encode_amd_df_config(0x1f, 0xffe), 0x0f00_fe1f);
        assert_eq!(encode_amd_df_config(0x2df, 0xfff), 0x020f_00ffdf);
    }

    #[test]
    fn grbm2_uses_arch_specific_tcp_and_rlc_bits() {
        assert!(grbm2_index(ChipClass::Gfx10).contains(&("RLC", 24)));
        assert!(grbm2_index(ChipClass::Gfx10).contains(&("TCP", 25)));
        assert!(grbm2_index(ChipClass::Gfx10_3).contains(&("RLC", 26)));
        assert!(grbm2_index(ChipClass::Gfx10_3).contains(&("TCP", 27)));
        assert!(grbm2_index(ChipClass::Gfx12).contains(&("RLC", 26)));
        assert!(grbm2_index(ChipClass::Gfx12).contains(&("TCP", 27)));
    }

    #[test]
    fn gpu_block_labels_match_requested_rows() {
        assert_eq!(
            GPU_BLOCK_LABELS,
            [
                "GFX Pipe",
                "Tex Pipe",
                "SPI",
                "RLC",
                "TCP",
                "UTCL2",
                "EA",
                "CP Fetch",
                "CP Compute",
            ]
        );
    }

    #[test]
    fn metric_header_and_rows_share_fixed_columns() {
        assert_eq!(METRIC_LABEL_WIDTH, 10);
        assert_eq!(metric_visual_width(80, METRIC_LABEL_WIDTH), 69);
        assert_eq!(metric_visual_width(160, METRIC_LABEL_WIDTH), 149);
        assert!(GPU_BLOCK_LABELS
            .iter()
            .all(|label| label.len() <= METRIC_LABEL_WIDTH));
        assert_eq!(
            metric_header_text(48, METRIC_LABEL_WIDTH, MetricView::Trend)
                .find("trend")
                .unwrap(),
            METRIC_LABEL_WIDTH + 1
        );
        let line = metric_line(
            "00",
            &Meter::new("00"),
            48,
            METRIC_LABEL_WIDTH,
            MetricView::Trend,
        );
        let sparkline_start =
            line.spans[0].content.chars().count() + line.spans[1].content.chars().count();
        assert_eq!(sparkline_start, METRIC_LABEL_WIDTH + 1);
    }

    #[test]
    fn visual_metric_view_contains_no_numeric_columns() {
        let mut meter = Meter::new("00");
        let now = Instant::now();
        meter.update(12.0, now, Duration::from_millis(800));
        meter.update(
            56.0,
            now + Duration::from_millis(800),
            Duration::from_millis(800),
        );
        meter.average_60s = 34.0;
        let line = metric_line("00", &meter, 32, METRIC_LABEL_WIDTH, MetricView::Trend);
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(SPARK_CHARS.iter().any(|ch| text.contains(*ch)));
        assert!(!text.contains("12.0%"));
        assert!(!text.contains("34.0%"));
        assert!(!text.contains("56.0%"));
        assert!(!metric_header_text(32, METRIC_LABEL_WIDTH, MetricView::Trend).contains("cur"));
    }

    #[test]
    fn text_metric_view_uses_fixed_width_numeric_columns() {
        let mut meter = Meter::new("00");
        meter.current = 12.0;
        meter.average_60s = 34.0;
        meter.peak = 56.0;
        let line = metric_line("00", &meter, 32, METRIC_LABEL_WIDTH, MetricView::Text);
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(text.chars().count(), 32);
        assert!(text.contains("12.0%"));
        assert!(text.contains("34.0%"));
        assert!(text.contains("56.0%"));
        assert_eq!(line.spans[1].style.fg, Some(heat_color(12.0)));
        assert_eq!(line.spans[2].style.fg, Some(heat_color(34.0)));
        assert_eq!(line.spans[3].style.fg, Some(heat_color(56.0)));
    }

    #[test]
    fn raw_text_metric_view_colors_values_by_normalized_scale() {
        let mut meter = RawMeter::new("pwr");
        meter.current = 25.0;
        meter.average_60s = 50.0;
        meter.peak = 100.0;
        meter.scale = 100.0;
        let line = raw_metric_line("pwr", &meter, 36, METRIC_LABEL_WIDTH, "W", MetricView::Text);
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(text.chars().count(), 36);
        assert!(text.contains("25.0W"));
        assert!(text.contains("50.0W"));
        assert!(text.contains("100.0W"));
        assert_eq!(line.spans[1].style.fg, Some(heat_color(25.0)));
        assert_eq!(line.spans[2].style.fg, Some(heat_color(50.0)));
        assert_eq!(line.spans[3].style.fg, Some(heat_color(100.0)));
    }

    #[test]
    fn cpu_panel_titles_and_rows_match_requested_modes() {
        let mut meter = Meter::new("00");
        meter.current = 100.0;
        meter.average_60s = 100.0;
        meter.peak = 100.0;
        assert_eq!(cpu_panel_title(MetricView::Text), "CPU cur  avg peak");
        assert_eq!(cpu_panel_title(MetricView::Trend), "CPU Utilization");
        let line_width = 24;

        let text_line = cpu_metric_line(&meter, line_width, MetricView::Text);
        let text = text_line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(text, "       00 100% 100% 100%");
        assert_eq!(text_line.spans[3].style.fg, Some(RED));
        assert_eq!(text_line.spans[5].style.fg, Some(RED));
        assert_eq!(text_line.spans[7].style.fg, Some(RED));

        let trend = cpu_metric_line(&meter, line_width, MetricView::Trend)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(trend.starts_with("00 "));
        assert!(trend.contains('▇'));
    }

    #[test]
    fn z_toggles_metric_view_between_trend_and_text() {
        let mut monitor = MonitorState::with_interval(Duration::from_secs(1));
        monitor.metric_view = MetricView::Trend;
        assert!(!handle_key(
            &mut monitor,
            KeyEvent::new_with_kind(
                KeyCode::Char('z'),
                crossterm::event::KeyModifiers::empty(),
                KeyEventKind::Press,
            )
        ));
        assert_eq!(monitor.metric_view, MetricView::Text);
        assert!(!handle_key(
            &mut monitor,
            KeyEvent::new_with_kind(
                KeyCode::Char('z'),
                crossterm::event::KeyModifiers::empty(),
                KeyEventKind::Press,
            )
        ));
        assert_eq!(monitor.metric_view, MetricView::Trend);
    }

    #[test]
    fn tab_and_number_keys_switch_monitor_views() {
        let mut monitor = MonitorState::with_interval(Duration::from_secs(1));
        monitor.view = MonitorView::Overview;
        assert!(!handle_key(
            &mut monitor,
            KeyEvent::new(KeyCode::Tab, crossterm::event::KeyModifiers::empty())
        ));
        assert_eq!(monitor.view, MonitorView::Compute);
        assert!(!handle_key(
            &mut monitor,
            KeyEvent::new(KeyCode::Char('1'), crossterm::event::KeyModifiers::empty())
        ));
        assert_eq!(monitor.view, MonitorView::Overview);
        assert!(!handle_key(
            &mut monitor,
            KeyEvent::new(KeyCode::Char('2'), crossterm::event::KeyModifiers::empty())
        ));
        assert_eq!(monitor.view, MonitorView::Compute);
    }

    #[test]
    fn terminal_color_classifier_respects_no_color() {
        assert_eq!(
            classify_terminal_color(
                Some("1"),
                Some("truecolor"),
                Some("xterm-256color"),
                Some(256)
            ),
            TerminalColorMode::Disabled
        );
    }

    #[test]
    fn terminal_color_classifier_detects_truecolor() {
        assert_eq!(
            classify_terminal_color(None, Some("truecolor"), Some("xterm-256color"), Some(256)),
            TerminalColorMode::TrueColor
        );
        assert_eq!(
            classify_terminal_color(None, None, Some("xterm-direct"), Some(256)),
            TerminalColorMode::TrueColor
        );
    }

    #[test]
    fn terminal_color_classifier_detects_ansi_levels() {
        assert_eq!(
            classify_terminal_color(None, None, Some("screen-256color"), None),
            TerminalColorMode::Ansi256
        );
        assert_eq!(
            classify_terminal_color(None, None, Some("vt100"), Some(8)),
            TerminalColorMode::Ansi16
        );
    }

    #[test]
    fn terminal_color_classifier_handles_dumb_and_unknown() {
        assert_eq!(
            classify_terminal_color(None, None, Some("dumb"), None),
            TerminalColorMode::Monochrome
        );
        assert_eq!(
            classify_terminal_color(None, None, None, None),
            TerminalColorMode::Unknown
        );
    }

    #[test]
    fn parses_swap_memory_from_meminfo() {
        let swap = parse_swap_memory(
            "\
MemTotal:       1000 kB
SwapTotal:      4096 kB
SwapFree:       1024 kB
",
        )
        .unwrap();
        assert_eq!(swap.total_bytes, 4096 * 1024);
        assert_eq!(swap.used_bytes, 3072 * 1024);
    }

    #[test]
    fn parses_swap_vmstat_counters() {
        let swap = parse_vmstat_swap(
            "\
pgpgin 1
pswpin 7
pswpout 11
pgpgout 2
",
        )
        .unwrap();
        assert_eq!(swap.pswpin, 7);
        assert_eq!(swap.pswpout, 11);
    }

    #[test]
    fn swap_thrash_requires_swap_used_and_paging_delta() {
        let mut sampler = SwapSampler {
            prev: Some(VmstatSwap {
                pswpin: 10,
                pswpout: 20,
            }),
        };
        let current = Some(VmstatSwap {
            pswpin: 16,
            pswpout: 28,
        });
        let pages_per_sec = match (sampler.prev, current) {
            (Some(prev), Some(next)) => {
                let delta = next
                    .pswpin
                    .saturating_sub(prev.pswpin)
                    .saturating_add(next.pswpout.saturating_sub(prev.pswpout));
                delta as f64 / 2.0
            }
            _ => 0.0,
        };
        sampler.prev = current;
        let alert = SwapThrash {
            active: 4096 > 0 && pages_per_sec > 0.0,
            pages_per_sec,
            used_bytes: 4096,
            total_bytes: 8192,
        };
        assert!(alert.active);
        assert_eq!(alert.pages_per_sec, 7.0);
        assert_eq!(sampler.prev.unwrap().pswpout, 28);
    }

    #[test]
    fn parses_network_counters_and_computes_reset_safe_rates() {
        let previous = parse_network_counters(
            "Inter-| Receive | Transmit\n eth0: 1000 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0\n lo: 99 0 0 0 0 0 0 0 99 0 0 0 0 0 0 0\n",
        );
        let current = parse_network_counters(
            "eth0: 5096 0 0 0 0 0 0 0 4048 0 0 0 0 0 0 0\n wlan0: 200 0 0 0 0 0 0 0 100 0 0 0 0 0 0 0\n",
        );
        let rates = network_rates(&previous, &current, Duration::from_secs(2));
        assert_eq!(rates.len(), 2);
        let eth0 = rates.iter().find(|rate| rate.name == "eth0").unwrap();
        assert_eq!(eth0.receive_bytes_per_sec, 2048.0);
        assert_eq!(eth0.transmit_bytes_per_sec, 1024.0);
        let wlan0 = rates.iter().find(|rate| rate.name == "wlan0").unwrap();
        assert_eq!(wlan0.receive_bytes_per_sec, 0.0);
        assert!(rates.iter().all(|rate| rate.name != "lo"));
    }

    #[test]
    fn parses_local_and_network_mounts_with_escaped_paths() {
        let mounts = parse_mountinfo(
            "36 25 0:31 / / rw,relatime - ext4 /dev/nvme0n1p2 rw\n37 25 0:32 / /mnt/model\\040cache rw,relatime - nfs4 server:/models rw\n38 25 0:33 / /proc rw,nosuid - proc proc rw\n",
        );
        assert_eq!(mounts.len(), 3);
        assert_eq!(mounts[1].mount_point, Path::new("/mnt/model cache"));
        assert!(is_operator_filesystem(&mounts[0]));
        assert!(is_operator_filesystem(&mounts[1]));
        assert!(!is_operator_filesystem(&mounts[2]));
    }

    #[test]
    fn overview_layout_adapts_without_overlapping_panels() {
        let wide = overview_areas(Rect::new(0, 0, 140, 36));
        assert_eq!(wide.host.y, wide.network.y);
        assert!(wide.network.x >= wide.host.x + wide.host.width);
        assert!(wide.clients.width > 0);
        assert!(wide.gpus.y >= wide.host.y + wide.host.height);

        let narrow = overview_areas(Rect::new(0, 0, 80, 36));
        assert_eq!(narrow.clients, Rect::default());
        assert!(narrow.network.y >= narrow.host.y + narrow.host.height);
        assert!(narrow.gpus.y >= narrow.network.y + narrow.network.height);
    }

    #[test]
    fn compute_layout_keeps_counters_and_accelerators_distinct() {
        let wide = compute_areas(Rect::new(0, 0, 140, 36));
        assert!(wide.gpu_blocks.x >= wide.cpu.x + wide.cpu.width);
        assert!(wide.gpus.x >= wide.gpu_blocks.x + wide.gpu_blocks.width);
        assert!(wide.npus.y >= wide.gpus.y + wide.gpus.height);

        let narrow = compute_areas(Rect::new(0, 0, 90, 36));
        assert_eq!(narrow.gpus.y, narrow.npus.y);
        assert!(narrow.cpu.y >= narrow.gpus.y + narrow.gpus.height);
        assert!(narrow.gpu_blocks.x >= narrow.cpu.x + narrow.cpu.width);
    }

    #[test]
    fn rate_and_capacity_formatters_remain_compact() {
        assert_eq!(format_rate(0.0), "0 B/s");
        assert_eq!(format_rate(2048.0), "2.0 KiB/s");
        assert_eq!(format_rate(5.0 * 1024.0 * 1024.0), "5.0 MiB/s");
        assert_eq!(truncate("very-long-interface", 8), "very-lo…");
    }

    #[test]
    fn npu_columns_fold_to_eight_percent_meters() {
        let stats = AdminStats {
            npus: vec![hipfire_admin_types::NpuTelemetry {
                columns_pct: vec![1, 2, 3, 4, 5, 6, 7, 250, 99],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(fold_npu_columns(&stats), vec![1, 2, 3, 4, 5, 6, 7, 100]);
    }

    #[test]
    fn perf_event_attr_uses_kernel_v0_prefix_size() {
        assert_eq!(std::mem::size_of::<PerfEventAttr>(), 64);
    }
}

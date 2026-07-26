// SPDX-License-Identifier: Apache-2.0
// hipfire — environment doctor and repair front-end.

use clap::Args;
use hipfire_config::{configured_models_dir, hipfire_dir, LoadedConfig};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire doctor\n  hipfire doctor --json\n  hipfire doctor --fix\n"
)]
pub struct DoctorArgs {
    /// Apply safe user-space fixes and invoke hipfire-priv-helper for privileged fixes.
    #[arg(long)]
    pub fix: bool,
    /// Emit the full report as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum CheckStatus {
    Pass,
    Warn,
    Fail,
    Info,
}

#[derive(Debug, Serialize)]
struct DoctorCheck {
    id: &'static str,
    status: CheckStatus,
    message: String,
    details: Value,
    fix: Option<String>,
}

#[derive(Debug, Serialize)]
struct DoctorFix {
    id: String,
    ok: bool,
    message: String,
    details: Value,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    ok: bool,
    fix_mode: bool,
    daemon_started: bool,
    checks: Vec<DoctorCheck>,
    fixes: Vec<DoctorFix>,
}

pub async fn run(args: DoctorArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    let mut report = DoctorReport {
        ok: false,
        fix_mode: args.fix,
        daemon_started: false,
        checks: Vec::new(),
        fixes: Vec::new(),
    };

    let mut ctx = DoctorContext::default();
    check_paths(&mut report, &loaded, args.fix);
    check_device_nodes(&mut report);
    check_npu_access(&mut report);
    check_gpu_driver_firmware(&mut report);
    check_npu_driver_firmware(&mut report);
    check_sysinfo(&mut report);
    check_hip_runtime(&mut report, &mut ctx);
    check_gfx1103_cwsr(&mut report, ctx.arch.as_deref());
    check_rocm_tools(&mut report);
    check_kernel_cache(&mut report, &loaded, ctx.arch.as_deref());
    report.daemon_started = check_lock(&mut report);
    if report.daemon_started {
        check_daemon_health(&mut report, &loaded).await;
    }
    check_monitoring_prereqs(&mut report, args.fix);

    report.ok = report
        .checks
        .iter()
        .all(|check| !matches!(check.status, CheckStatus::Fail));

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }
    if report.ok {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

#[derive(Default)]
struct DoctorContext {
    arch: Option<String>,
}

fn check_paths(report: &mut DoctorReport, loaded: &LoadedConfig, fix: bool) {
    let root = hipfire_dir();
    let models = configured_models_dir(&loaded.config);
    let bin = root.join("bin");
    let locks = root.join("locks");
    for (id, path, label) in [
        ("path.hipfire_dir", root, "hipfire directory"),
        ("path.models_dir", models, "models directory"),
        ("path.bin_dir", bin, "binary directory"),
        ("path.locks_dir", locks, "resource lock directory"),
    ] {
        let exists = path.is_dir();
        if !exists && fix {
            match fs::create_dir_all(&path) {
                Ok(()) => report.fixes.push(DoctorFix {
                    id: format!("{id}.create"),
                    ok: true,
                    message: format!("created {label}: {}", path.display()),
                    details: json!({ "path": path }),
                }),
                Err(err) => report.fixes.push(DoctorFix {
                    id: format!("{id}.create"),
                    ok: false,
                    message: format!("failed to create {label}: {err}"),
                    details: json!({ "path": path }),
                }),
            }
        }
        let exists_after = path.is_dir();
        report.checks.push(DoctorCheck {
            id,
            status: if exists_after {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail
            },
            message: if exists_after {
                format!("{label} exists: {}", path.display())
            } else {
                format!("{label} is missing: {}", path.display())
            },
            details: json!({ "path": path, "exists": exists_after }),
            fix: (!exists_after).then(|| format!("create {}", path.display())),
        });
    }
}

fn check_device_nodes(report: &mut DoctorReport) {
    let kfd = Path::new("/dev/kfd");
    let kfd_ok = kfd.exists() && can_read_write(kfd);
    report.checks.push(DoctorCheck {
        id: "device.kfd",
        status: if kfd_ok {
            CheckStatus::Pass
        } else if kfd.exists() {
            CheckStatus::Fail
        } else {
            CheckStatus::Fail
        },
        message: if kfd_ok {
            "/dev/kfd is present and readable/writable".to_string()
        } else if kfd.exists() {
            "/dev/kfd exists but is not readable/writable by this user".to_string()
        } else {
            "/dev/kfd is missing".to_string()
        },
        details: json!({ "path": "/dev/kfd", "exists": kfd.exists(), "read_write": kfd_ok }),
        fix: (!kfd_ok).then(|| {
            "install ROCm driver stack and ensure the user has render/video group access"
                .to_string()
        }),
    });

    let render_nodes = render_nodes();
    let accessible = render_nodes
        .iter()
        .filter(|path| can_read_write(path))
        .count();
    let status = if accessible > 0 {
        CheckStatus::Pass
    } else if render_nodes.is_empty() {
        CheckStatus::Fail
    } else {
        CheckStatus::Fail
    };
    report.checks.push(DoctorCheck {
        id: "device.render",
        status,
        message: match (render_nodes.len(), accessible) {
            (0, _) => "no /dev/dri/renderD* nodes found".to_string(),
            (_, 0) => "render nodes exist but none are readable/writable by this user".to_string(),
            (total, ok) => format!("{ok}/{total} render nodes are readable/writable"),
        },
        details: json!({
            "nodes": render_nodes.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "accessible": accessible,
        }),
        fix: (accessible == 0).then(|| {
            "ensure /dev/dri/renderD* exists and the user has render group access".to_string()
        }),
    });
}

fn check_npu_access(report: &mut DoctorReport) {
    let nodes = accel_nodes();
    let node_details: Vec<_> = nodes
        .iter()
        .map(|path| {
            json!({
                "path": path,
                "read_write": can_read_write(path),
            })
        })
        .collect();
    let accessible = nodes.iter().filter(|path| can_read_write(path)).count();
    let open_result = match hipfire_xdna::XdnaDevice::open_default() {
        Ok(dev) => {
            let resource_info = dev.resource_info().map(|r| {
                json!({
                    "npu_clk_max": r.npu_clk_max,
                    "tops_max": r.npu_tops_max,
                    "tops_current": r.npu_tops_curr,
                    "tasks_max": r.npu_task_max,
                    "tasks_current": r.npu_task_curr,
                })
            });
            let clocks = dev.clocks().map(|c| {
                json!({
                    "mp_npu_mhz": c.mp_npu_mhz,
                    "h_mhz": c.h_mhz,
                })
            });
            let sensors = dev.sensors().map(|s| {
                json!({
                    "power_mw": s.power_mw,
                    "temp_c": s.temp_c,
                    "mean_util_pct": s.mean_utilization_pct(),
                    "columns_pct": s.column_utilization_pct,
                })
            });
            json!({
                "ok": true,
                "path": dev.path(),
                "resource_info": result_json(resource_info),
                "clocks": result_json(clocks),
                "sensors": result_json(sensors),
            })
        }
        Err(err) => json!({
            "ok": false,
            "error": err.to_string(),
        }),
    };
    let open_ok = open_result
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    report.checks.push(DoctorCheck {
        id: "access.npu",
        status: if nodes.is_empty() {
            CheckStatus::Info
        } else if accessible == 0 || !open_ok {
            CheckStatus::Fail
        } else {
            CheckStatus::Pass
        },
        message: match (nodes.len(), accessible, open_ok) {
            (0, _, _) => "no /dev/accel/accel* NPU nodes found".to_string(),
            (total, ok, true) => {
                format!("{ok}/{total} NPU accel nodes are readable/writable; XDNA ioctl open works")
            }
            (total, ok, false) => format!(
                "{ok}/{total} NPU accel nodes are readable/writable, but XDNA ioctl open failed"
            ),
        },
        details: json!({
            "nodes": node_details,
            "accessible": accessible,
            "xdna_open": open_result,
        }),
        fix: (!nodes.is_empty() && (accessible == 0 || !open_ok)).then(|| {
            "ensure /dev/accel/accel* exists and the user has render group access".to_string()
        }),
    });
}

fn check_gpu_driver_firmware(report: &mut DoctorReport) {
    let module = kernel_module_info("amdgpu");
    let cards = amd_drm_cards();
    report.checks.push(DoctorCheck {
        id: "driver.gpu",
        status: if !module.exists {
            CheckStatus::Fail
        } else if cards.is_empty() {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if !module.exists {
            "amdgpu kernel module is not loaded".to_string()
        } else if cards.is_empty() {
            "amdgpu module is loaded, but no AMD DRM cards were found".to_string()
        } else {
            let src = module
                .srcversion
                .as_deref()
                .map(|v| format!(" srcversion {v}"))
                .unwrap_or_default();
            format!(
                "amdgpu module is loaded for {} AMD DRM card(s){src}",
                cards.len()
            )
        },
        details: json!({
            "module": module,
            "cards": cards.iter().map(gpu_card_json).collect::<Vec<_>>(),
        }),
        fix: (!module.exists)
            .then(|| "load the amdgpu kernel module / install the AMD GPU driver".to_string()),
    });

    let firmware_cards: Vec<_> = cards
        .iter()
        .map(|card| {
            json!({
                "card": card.card,
                "device": card.device_path,
                "vbios_version": card.vbios_version,
                "gpu_metrics_version": card.gpu_metrics_version,
                "firmware": card.firmware,
            })
        })
        .collect();
    let firmware_count: usize = cards.iter().map(|card| card.firmware.len()).sum();
    report.checks.push(DoctorCheck {
        id: "firmware.gpu",
        status: if cards.is_empty() {
            CheckStatus::Warn
        } else if firmware_count == 0 {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if cards.is_empty() {
            "GPU firmware check skipped because no AMD DRM cards were found".to_string()
        } else if firmware_count == 0 {
            "AMD DRM card(s) found, but no fw_version entries were exposed".to_string()
        } else {
            format!(
                "GPU firmware versions visible: {firmware_count} component(s) across {} card(s)",
                cards.len()
            )
        },
        details: json!({ "cards": firmware_cards }),
        fix: None,
    });
}

fn check_npu_driver_firmware(report: &mut DoctorReport) {
    let module = kernel_module_info("amdxdna");
    let devices = accel_sysfs_devices();
    let device_ids: Vec<_> = devices
        .iter()
        .filter_map(|device| device.uevent.get("PCI_ID").cloned())
        .collect();
    report.checks.push(DoctorCheck {
        id: "driver.npu",
        status: if devices.is_empty() && !module.exists {
            CheckStatus::Info
        } else if !module.exists {
            CheckStatus::Fail
        } else if devices.is_empty() {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if devices.is_empty() && !module.exists {
            "no XDNA NPU device or amdxdna module found".to_string()
        } else if !module.exists {
            "XDNA accel device exists, but the amdxdna kernel module is not loaded".to_string()
        } else if devices.is_empty() {
            "amdxdna module is loaded, but no /sys/class/accel devices were found".to_string()
        } else {
            let version = module
                .version
                .as_deref()
                .or(module.srcversion.as_deref())
                .unwrap_or("unknown version");
            format!(
                "amdxdna driver {version} sees {} accel device(s)",
                devices.len()
            )
        },
        details: json!({
            "module": module,
            "devices": devices,
        }),
        fix: (!devices.is_empty() && !module.exists)
            .then(|| "load the amdxdna kernel module / install the XDNA NPU driver".to_string()),
    });

    let firmware = installed_npu_firmware_candidates(&device_ids);
    report.checks.push(DoctorCheck {
        id: "firmware.npu",
        status: if devices.is_empty() && firmware.is_empty() {
            CheckStatus::Info
        } else if firmware.is_empty() {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if devices.is_empty() && firmware.is_empty() {
            "NPU firmware check skipped because no XDNA NPU device was found".to_string()
        } else if firmware.is_empty() {
            "no installed amdnpu firmware candidates found for detected NPU PCI IDs".to_string()
        } else {
            format!(
                "installed NPU firmware candidates found for detected PCI ID(s): {} file(s)",
                firmware.len()
            )
        },
        details: json!({
            "pci_ids": device_ids,
            "installed_candidates": firmware,
            "loaded_firmware_version_exposed": false,
            "note": "amdxdna does not expose a loaded NPU firmware version through the sysfs nodes checked here",
        }),
        fix: None,
    });
}

#[derive(Debug, Serialize)]
struct KernelModuleInfo {
    name: String,
    exists: bool,
    initstate: Option<String>,
    version: Option<String>,
    srcversion: Option<String>,
}

#[derive(Debug)]
struct GpuCardInfo {
    card: String,
    device_path: String,
    vendor_id: Option<String>,
    device_id: Option<String>,
    revision: Option<String>,
    subsystem_vendor: Option<String>,
    subsystem_device: Option<String>,
    driver: Option<String>,
    pci_slot_name: Option<String>,
    vbios_version: Option<String>,
    gpu_metrics_version: Option<String>,
    firmware: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct AccelSysfsDevice {
    accel: String,
    sysfs_path: String,
    devnode: Option<String>,
    devnode_read_write: Option<bool>,
    uevent: BTreeMap<String, String>,
    power_state: Option<String>,
    runtime_status: Option<String>,
}

fn kernel_module_info(name: &str) -> KernelModuleInfo {
    let root = Path::new("/sys/module").join(name);
    KernelModuleInfo {
        name: name.to_string(),
        exists: root.exists(),
        initstate: read_trimmed(&root.join("initstate")),
        version: read_trimmed(&root.join("version")),
        srcversion: read_trimmed(&root.join("srcversion")),
    }
}

fn amd_drm_cards() -> Vec<GpuCardInfo> {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut cards = Vec::new();
    for entry in entries.flatten() {
        let card = entry.file_name().to_string_lossy().to_string();
        if !is_card_name(&card) {
            continue;
        }
        let device = entry.path().join("device");
        if read_trimmed(&device.join("vendor")).as_deref() != Some("0x1002") {
            continue;
        }
        let uevent = uevent_map(&device.join("uevent"));
        let metrics_version = hipfire_sysinfo::read_gpu_metrics(&device)
            .map(|m| format!("{}.{}", m.version.0, m.version.1));
        cards.push(GpuCardInfo {
            card,
            device_path: device.display().to_string(),
            vendor_id: read_trimmed(&device.join("vendor")),
            device_id: read_trimmed(&device.join("device")),
            revision: read_trimmed(&device.join("revision")),
            subsystem_vendor: read_trimmed(&device.join("subsystem_vendor")),
            subsystem_device: read_trimmed(&device.join("subsystem_device")),
            driver: uevent.get("DRIVER").cloned(),
            pci_slot_name: uevent.get("PCI_SLOT_NAME").cloned(),
            vbios_version: read_trimmed(&device.join("vbios_version")),
            gpu_metrics_version: metrics_version,
            firmware: firmware_versions(&device),
        });
    }
    cards.sort_by(|a, b| a.card.cmp(&b.card));
    cards
}

fn gpu_card_json(card: &GpuCardInfo) -> Value {
    json!({
        "card": card.card,
        "device_path": card.device_path,
        "vendor_id": card.vendor_id,
        "device_id": card.device_id,
        "revision": card.revision,
        "subsystem_vendor": card.subsystem_vendor,
        "subsystem_device": card.subsystem_device,
        "driver": card.driver,
        "pci_slot_name": card.pci_slot_name,
        "vbios_version": card.vbios_version,
        "gpu_metrics_version": card.gpu_metrics_version,
        "firmware": card.firmware,
    })
}

fn firmware_versions(device: &Path) -> BTreeMap<String, String> {
    let mut versions = BTreeMap::new();
    let fw_dir = device.join("fw_version");
    let Ok(entries) = fs::read_dir(fw_dir) else {
        return versions;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(value) = read_trimmed(&path) {
            versions.insert(name.to_string(), value);
        }
    }
    versions
}

fn accel_sysfs_devices() -> Vec<AccelSysfsDevice> {
    let Ok(entries) = fs::read_dir("/sys/class/accel") else {
        return Vec::new();
    };
    let mut devices = Vec::new();
    for entry in entries.flatten() {
        let accel = entry.file_name().to_string_lossy().to_string();
        if !is_accel_name(&accel) {
            continue;
        }
        let root = entry.path();
        let class_uevent = uevent_map(&root.join("uevent"));
        let device_uevent = uevent_map(&root.join("device").join("uevent"));
        let mut uevent = class_uevent;
        uevent.extend(device_uevent);
        let devnode = uevent.get("DEVNAME").map(|name| format!("/dev/{name}"));
        let devnode_read_write = devnode
            .as_deref()
            .map(|node| can_read_write(Path::new(node)));
        devices.push(AccelSysfsDevice {
            accel,
            sysfs_path: root.display().to_string(),
            devnode,
            devnode_read_write,
            uevent,
            power_state: read_trimmed(&root.join("device").join("power_state")),
            runtime_status: read_trimmed(&root.join("power").join("runtime_status")),
        });
    }
    devices.sort_by(|a, b| a.accel.cmp(&b.accel));
    devices
}

fn installed_npu_firmware_candidates(pci_ids: &[String]) -> Vec<Value> {
    let mut candidates = BTreeMap::new();
    for pci_id in pci_ids {
        let Some((_, device_id)) = pci_id.split_once(':') else {
            continue;
        };
        let device_id = device_id.to_ascii_lowercase();
        for root in [
            "/lib/firmware/updates/amdnpu",
            "/lib/firmware/amdnpu",
            "/usr/lib/firmware/updates/amdnpu",
            "/usr/lib/firmware/amdnpu",
        ] {
            let Ok(dirs) = fs::read_dir(root) else {
                continue;
            };
            for dir in dirs.flatten() {
                let dir_path = dir.path();
                if !dir_path.is_dir() {
                    continue;
                }
                let Some(dir_name) = dir_path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !dir_name
                    .to_ascii_lowercase()
                    .starts_with(&format!("{device_id}_"))
                {
                    continue;
                }
                let Ok(files) = fs::read_dir(&dir_path) else {
                    continue;
                };
                for file in files.flatten() {
                    let file_path = file.path();
                    if !file_path.is_file() {
                        continue;
                    }
                    let Some(file_name) = file_path.file_name().and_then(|name| name.to_str())
                    else {
                        continue;
                    };
                    if !file_name.starts_with("npu") && !file_name.starts_with("cert") {
                        continue;
                    }
                    let key = file_path
                        .canonicalize()
                        .unwrap_or_else(|_| file_path.clone())
                        .display()
                        .to_string();
                    candidates.insert(
                        key,
                        json!({
                            "pci_id": pci_id,
                            "family": dir_name,
                            "file": file_name,
                            "version": npu_firmware_version_from_name(file_name),
                            "path": file_path,
                        }),
                    );
                }
            }
        }
    }
    candidates.into_values().collect()
}

fn npu_firmware_version_from_name(name: &str) -> Option<String> {
    let name = name.strip_suffix(".zst").unwrap_or(name);
    name.strip_prefix("npu.sbin.").map(ToString::to_string)
}

fn uevent_map(path: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Some(text) = read_trimmed(path) else {
        return out;
    };
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        out.insert(key.to_string(), value.to_string());
    }
    out
}

fn result_json<T, E>(result: Result<T, E>) -> Value
where
    T: Serialize,
    E: ToString,
{
    match result {
        Ok(value) => json!({ "ok": true, "value": value }),
        Err(err) => json!({ "ok": false, "error": err.to_string() }),
    }
}

fn check_sysinfo(report: &mut DoctorReport) {
    let snapshot = hipfire_sysinfo::snapshot(now_unix());
    report.checks.push(DoctorCheck {
        id: "sysinfo.gpus",
        status: if snapshot.gpus.is_empty() {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if snapshot.gpus.is_empty() {
            "sysfs telemetry found no AMD GPUs".to_string()
        } else {
            format!("sysfs telemetry sees {} AMD GPU(s)", snapshot.gpus.len())
        },
        details: serde_json::to_value(&snapshot.gpus).unwrap_or_else(|_| json!([])),
        fix: None,
    });
}

fn check_hip_runtime(report: &mut DoctorReport, ctx: &mut DoctorContext) {
    match hip_bridge::HipRuntime::load() {
        Ok(hip) => {
            let version = hip.runtime_version().ok();
            let device_count = hip.device_count().ok();
            let arch = if device_count.unwrap_or(0) > 0 {
                hip.get_arch(0).ok()
            } else {
                None
            };
            ctx.arch = arch.clone();
            report.checks.push(DoctorCheck {
                id: "hip.runtime",
                status: if device_count.unwrap_or(0) > 0 {
                    CheckStatus::Pass
                } else {
                    CheckStatus::Fail
                },
                message: match (version, device_count, arch.as_deref()) {
                    (Some((maj, min)), Some(n), Some(arch)) => {
                        format!("HIP runtime {maj}.{min} sees {n} device(s), device 0 is {arch}")
                    }
                    (Some((maj, min)), Some(n), None) => {
                        format!("HIP runtime {maj}.{min} sees {n} device(s)")
                    }
                    (_, Some(n), _) => format!("HIP runtime loaded and sees {n} device(s)"),
                    _ => "HIP runtime loaded but device count failed".to_string(),
                },
                details: json!({
                    "runtime_version": version.map(|(major, minor)| json!({ "major": major, "minor": minor })),
                    "device_count": device_count,
                    "arch": arch,
                }),
                fix: (device_count.unwrap_or(0) <= 0)
                    .then(|| "check ROCm install and device permissions".to_string()),
            });
        }
        Err(err) => report.checks.push(DoctorCheck {
            id: "hip.runtime",
            status: CheckStatus::Fail,
            message: format!("failed to load HIP runtime: {err}"),
            details: json!({ "error": err.to_string() }),
            fix: Some("fix libamdhip64.so resolution or install ROCm runtime".to_string()),
        }),
    }
}

const CWSR_ENABLE_PATH: &str = "/sys/module/amdgpu/parameters/cwsr_enable";

fn check_gfx1103_cwsr(report: &mut DoctorReport, arch: Option<&str>) {
    let Some(arch) = arch else {
        return;
    };
    let value = fs::read_to_string(CWSR_ENABLE_PATH)
        .map(|value| value.trim().to_string())
        .map_err(|err| err.to_string());
    if let Some(check) = build_gfx1103_cwsr_check(arch, value) {
        report.checks.push(check);
    }
}

fn build_gfx1103_cwsr_check(arch: &str, value: Result<String, String>) -> Option<DoctorCheck> {
    if arch.split(':').next() != Some("gfx1103") {
        return None;
    }

    let remedy =
        "set the kernel command line to amdgpu.cwsr_enable=0, reboot, and rerun hipfire doctor";
    Some(match value {
        Ok(value) => match parse_module_bool(&value) {
            Some(false) => DoctorCheck {
                id: "driver.gpu_cwsr",
                status: CheckStatus::Pass,
                message: "gfx1103 CWSR is off; the LDS/preemption workaround is active"
                    .to_string(),
                details: json!({
                    "arch": arch,
                    "path": CWSR_ENABLE_PATH,
                    "value": value,
                    "enabled": false,
                    "workaround_active": true,
                }),
                fix: None,
            },
            Some(true) => DoctorCheck {
                id: "driver.gpu_cwsr",
                status: CheckStatus::Warn,
                message: "gfx1103 CWSR is on; multi-wave barrier/LDS workloads can hang during preemption"
                    .to_string(),
                details: json!({
                    "arch": arch,
                    "path": CWSR_ENABLE_PATH,
                    "value": value,
                    "enabled": true,
                    "workaround_active": false,
                }),
                fix: Some(remedy.to_string()),
            },
            None => DoctorCheck {
                id: "driver.gpu_cwsr",
                status: CheckStatus::Warn,
                message: format!(
                    "gfx1103 CWSR state is unrecognized ({value:?}); the LDS/preemption workaround cannot be verified"
                ),
                details: json!({
                    "arch": arch,
                    "path": CWSR_ENABLE_PATH,
                    "value": value,
                    "enabled": null,
                    "workaround_active": false,
                }),
                fix: Some(remedy.to_string()),
            },
        },
        Err(err) => DoctorCheck {
            id: "driver.gpu_cwsr",
            status: CheckStatus::Warn,
            message: "gfx1103 CWSR state could not be read; the LDS/preemption workaround cannot be verified"
                .to_string(),
            details: json!({
                "arch": arch,
                "path": CWSR_ENABLE_PATH,
                "error": err,
                "enabled": null,
                "workaround_active": false,
            }),
            fix: Some(remedy.to_string()),
        },
    })
}

fn parse_module_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "y" | "yes" | "true" => Some(true),
        "0" | "n" | "no" | "false" => Some(false),
        _ => None,
    }
}

fn check_rocm_tools(report: &mut DoctorReport) {
    for tool in ["hipcc", "rocminfo", "rocm-smi"] {
        let path = find_in_path(tool);
        report.checks.push(DoctorCheck {
            id: match tool {
                "hipcc" => "tool.hipcc",
                "rocminfo" => "tool.rocminfo",
                _ => "tool.rocm_smi",
            },
            status: if path.is_some() {
                CheckStatus::Pass
            } else {
                CheckStatus::Warn
            },
            message: match &path {
                Some(path) => format!("{tool} found at {}", path.display()),
                None => format!("{tool} not found in PATH"),
            },
            details: json!({ "path": path }),
            fix: None,
        });
    }
}

fn check_kernel_cache(report: &mut DoctorReport, loaded: &LoadedConfig, arch: Option<&str>) {
    let Some(arch) = arch else {
        report.checks.push(DoctorCheck {
            id: "kernels.cache",
            status: CheckStatus::Warn,
            message: "kernel cache check skipped because HIP arch is unknown".to_string(),
            details: json!({}),
            fix: None,
        });
        return;
    };
    let root = hipfire_dir();
    let candidates = [
        std::env::var_os("HIPFIRE_KERNEL_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("kernels"))
            .join(arch),
        root.join("kernels").join("compiled").join(arch),
        workspace_root()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("kernels")
            .join("compiled")
            .join(arch),
    ];
    let existing: Vec<_> = candidates
        .iter()
        .filter(|path| path.is_dir())
        .map(|path| path.display().to_string())
        .collect();
    let model_dir = configured_models_dir(&loaded.config);
    report.checks.push(DoctorCheck {
        id: "kernels.cache",
        status: if existing.is_empty() {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        },
        message: if existing.is_empty() {
            format!("no precompiled kernel cache directory found for {arch}; JIT may be used")
        } else {
            format!("kernel cache candidate(s) found for {arch}")
        },
        details: json!({
            "arch": arch,
            "existing": existing,
            "models_dir": model_dir,
        }),
        fix: None,
    });
}

fn check_lock(report: &mut DoctorReport) -> bool {
    let path = hipfire_lock::gpu_resource_lock_path();
    match hipfire_lock::probe(&path) {
        Ok(hipfire_lock::LockState::Free) => {
            report.checks.push(DoctorCheck {
                id: "lock.gpu",
                status: CheckStatus::Pass,
                message: "GPU resource lock is free".to_string(),
                details: json!({ "path": path }),
                fix: None,
            });
            false
        }
        Ok(hipfire_lock::LockState::Busy(holder)) => {
            report.checks.push(DoctorCheck {
                id: "lock.gpu",
                status: CheckStatus::Info,
                message: format!("GPU resource lock is busy: {holder}"),
                details: json!({ "path": path, "holder": holder }),
                fix: None,
            });
            true
        }
        Err(err) => {
            report.checks.push(DoctorCheck {
                id: "lock.gpu",
                status: CheckStatus::Warn,
                message: format!("could not probe GPU resource lock: {err}"),
                details: json!({ "path": path, "error": err.to_string() }),
                fix: None,
            });
            false
        }
    }
}

async fn check_daemon_health(report: &mut DoctorReport, loaded: &LoadedConfig) {
    let host = match loaded.config.host.as_str() {
        "" | "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    let url = format!("http://{}:{}/health", host, loaded.config.port);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
    {
        Ok(client) => client,
        Err(err) => {
            report.checks.push(DoctorCheck {
                id: "daemon.health",
                status: CheckStatus::Warn,
                message: format!("could not create HTTP client: {err}"),
                details: json!({ "url": url }),
                fix: None,
            });
            return;
        }
    };
    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => report.checks.push(DoctorCheck {
            id: "daemon.health",
            status: CheckStatus::Pass,
            message: format!("server health endpoint is reachable at {url}"),
            details: json!({ "url": url, "status": response.status().as_u16() }),
            fix: None,
        }),
        Ok(response) => report.checks.push(DoctorCheck {
            id: "daemon.health",
            status: CheckStatus::Warn,
            message: format!("server health endpoint returned {}", response.status()),
            details: json!({ "url": url, "status": response.status().as_u16() }),
            fix: None,
        }),
        Err(err) => report.checks.push(DoctorCheck {
            id: "daemon.health",
            status: CheckStatus::Info,
            message: format!("server health endpoint is not reachable at {url}: {err}"),
            details: json!({ "url": url, "error": err.to_string() }),
            fix: None,
        }),
    }
}

fn check_monitoring_prereqs(report: &mut DoctorReport, fix: bool) {
    let helper = find_priv_helper();
    let helper_probe = helper.as_deref().map(probe_helper_direct);
    let helper_can_execute = helper.as_deref().is_some_and(is_executable);
    let helper_direct_root = helper_probe
        .as_ref()
        .and_then(|probe| probe.get("details"))
        .and_then(|details| details.get("is_root"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let pkexec = find_in_path("pkexec");
    report.checks.push(DoctorCheck {
        id: "priv_helper.binary",
        status: if helper_can_execute && (helper_direct_root || pkexec.is_some()) {
            CheckStatus::Pass
        } else if helper.is_some() {
            CheckStatus::Warn
        } else {
            CheckStatus::Warn
        },
        message: match (
            &helper,
            helper_can_execute,
            helper_direct_root,
            pkexec.as_ref(),
        ) {
            (Some(path), true, true, _) => format!(
                "hipfire-priv-helper found and already runs as root: {}",
                path.display()
            ),
            (Some(path), true, false, Some(pkexec)) => format!(
                "hipfire-priv-helper found at {}; polkit elevation available via {}",
                path.display(),
                pkexec.display()
            ),
            (Some(path), true, false, None) => format!(
                "hipfire-priv-helper found at {}, but pkexec is not available; sudo fallback will be printed",
                path.display()
            ),
            (Some(path), false, _, _) => {
                format!(
                    "hipfire-priv-helper found but is not executable: {}",
                    path.display()
                )
            }
            (None, _, _, _) => {
                "hipfire-priv-helper not found; privileged doctor fixes will be instructions only"
                    .to_string()
            }
        },
        details: json!({
            "path": helper,
            "metadata": helper.as_deref().and_then(helper_metadata_json),
            "direct_probe": helper_probe,
            "direct_effective_root": helper_direct_root,
            "pkexec": pkexec,
            "sudo_fallback": find_in_path("sudo"),
        }),
        fix: None,
    });

    let amd_uncore_loaded = Path::new("/sys/module/amd_uncore").exists();
    report.checks.push(DoctorCheck {
        id: "monitor.amd_uncore",
        status: if amd_uncore_loaded {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn
        },
        message: if amd_uncore_loaded {
            "amd_uncore module is loaded".to_string()
        } else {
            "amd_uncore module is not loaded; memory bandwidth counters may be unavailable"
                .to_string()
        },
        details: json!({ "loaded": amd_uncore_loaded }),
        fix: (!amd_uncore_loaded).then(|| {
            privileged_fix_command(
                helper.as_deref(),
                pkexec.as_deref(),
                &["load-module", "amd_uncore"],
            )
        }),
    });

    let resctrl = is_mountpoint("/sys/fs/resctrl");
    report.checks.push(DoctorCheck {
        id: "monitor.resctrl",
        status: if resctrl {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn
        },
        message: if resctrl {
            "resctrl is mounted".to_string()
        } else {
            "resctrl is not mounted; cache/memory QoS counters may be unavailable".to_string()
        },
        details: json!({ "mounted": resctrl, "path": "/sys/fs/resctrl" }),
        fix: (!resctrl).then(|| {
            privileged_fix_command(helper.as_deref(), pkexec.as_deref(), &["mount-resctrl"])
        }),
    });

    let paranoid = fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());
    report.checks.push(DoctorCheck {
        id: "monitor.perf_event_paranoid",
        status: match paranoid {
            Some(v) if v <= 1 => CheckStatus::Pass,
            Some(_) => CheckStatus::Warn,
            None => CheckStatus::Warn,
        },
        message: match paranoid {
            Some(v) if v <= 1 => format!("perf_event_paranoid={v} allows broad perf access"),
            Some(v) => format!("perf_event_paranoid={v}; some hardware counters may require root"),
            None => "could not read perf_event_paranoid".to_string(),
        },
        details: json!({ "value": paranoid }),
        fix: paranoid.filter(|v| *v > 1).map(|_| {
            privileged_fix_command(
                helper.as_deref(),
                pkexec.as_deref(),
                &["set-perf-event-paranoid", "1"],
            )
        }),
    });

    if fix {
        if let Some(helper) = helper {
            if !amd_uncore_loaded {
                report.fixes.push(run_helper(
                    &helper,
                    pkexec.as_deref(),
                    &["load-module", "amd_uncore"],
                ));
            }
            if !resctrl {
                report
                    .fixes
                    .push(run_helper(&helper, pkexec.as_deref(), &["mount-resctrl"]));
            }
            if paranoid.is_some_and(|v| v > 1) {
                report.fixes.push(run_helper(
                    &helper,
                    pkexec.as_deref(),
                    &["set-perf-event-paranoid", "1"],
                ));
            }
        } else {
            report.fixes.push(DoctorFix {
                id: "priv_helper.missing".to_string(),
                ok: false,
                message:
                    "hipfire-priv-helper is not installed; privileged fixes were not attempted"
                        .to_string(),
                details: json!({}),
            });
        }
    }
}

fn run_helper(helper: &Path, pkexec: Option<&Path>, args: &[&str]) -> DoctorFix {
    let id = format!("priv_helper.{}", args.join("."));
    let direct_probe = probe_helper_direct(helper);
    let direct_root = direct_probe
        .get("details")
        .and_then(|details| details.get("is_root"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut command = if direct_root {
        let mut command = Command::new(helper);
        command.args(args);
        command
    } else if let Some(pkexec) = pkexec {
        let mut command = Command::new(pkexec);
        command.arg(helper).args(args);
        command
    } else {
        return DoctorFix {
            id,
            ok: false,
            message: format!(
                "privileged fix needs polkit; run manually: {}",
                privileged_fix_command(Some(helper), None, args)
            ),
            details: json!({
                "helper": helper,
                "args": args,
                "direct_probe": direct_probe,
                "pkexec": null,
            }),
        };
    };
    let invoked_via = if direct_root { "direct" } else { "pkexec" };
    match command.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let parsed: Value = serde_json::from_str(&stdout).unwrap_or_else(|_| json!({}));
            let message = if output.status.success() {
                parsed
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("helper command succeeded")
                    .to_string()
            } else if !stderr.is_empty() {
                stderr.clone()
            } else {
                parsed
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("helper command failed")
                    .to_string()
            };
            DoctorFix {
                id,
                ok: output.status.success(),
                message,
                details: json!({
                    "helper": helper,
                    "args": args,
                    "invoked_via": invoked_via,
                    "direct_probe": direct_probe,
                    "pkexec": pkexec,
                    "status": output.status.code(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "json": parsed,
                }),
            }
        }
        Err(err) => DoctorFix {
            id,
            ok: false,
            message: format!("failed to run privileged helper via {invoked_via}: {err}"),
            details: json!({
                "helper": helper,
                "args": args,
                "invoked_via": invoked_via,
                "direct_probe": direct_probe,
                "pkexec": pkexec,
            }),
        },
    }
}

fn print_text_report(report: &DoctorReport) {
    println!("hipfire doctor");
    println!("fix mode: {}", if report.fix_mode { "on" } else { "off" });
    for check in &report.checks {
        let label = match check.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Warn => "warn",
            CheckStatus::Fail => "fail",
            CheckStatus::Info => "info",
        };
        println!("[{label}] {:28} {}", check.id, check.message);
        if let Some(fix) = &check.fix {
            println!("       fix: {fix}");
        }
    }
    if !report.fixes.is_empty() {
        println!();
        println!("fix results");
        for fix in &report.fixes {
            println!(
                "[{}] {:28} {}",
                if fix.ok { "pass" } else { "fail" },
                fix.id,
                fix.message
            );
        }
    }
    if !report.daemon_started {
        println!("daemon is not started");
    }
}

fn can_read_write(path: &Path) -> bool {
    let bytes = path.as_os_str().as_bytes();
    let Ok(c_path) = CString::new(bytes) else {
        return false;
    };
    unsafe { libc::access(c_path.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
}

fn render_nodes() -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir("/dev/dri") else {
        return Vec::new();
    };
    let mut nodes: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.strip_prefix("renderD")
                        .is_some_and(|suffix| suffix.chars().all(|c| c.is_ascii_digit()))
                })
        })
        .collect();
    nodes.sort();
    nodes
}

fn accel_nodes() -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir("/dev/accel") else {
        return Vec::new();
    };
    let mut nodes: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_accel_name)
        })
        .collect();
    nodes.sort();
    nodes
}

fn is_card_name(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
}

fn is_accel_name(name: &str) -> bool {
    name.strip_prefix("accel")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn helper_metadata_json(path: &Path) -> Option<Value> {
    let meta = fs::metadata(path).ok()?;
    let mode = meta.permissions().mode();
    Some(json!({
        "uid": meta.uid(),
        "gid": meta.gid(),
        "mode_octal": format!("{:04o}", mode & 0o7777),
        "executable": mode & 0o111 != 0,
        "setuid": mode & 0o4000 != 0,
        "group_writable": mode & 0o020 != 0,
        "world_writable": mode & 0o002 != 0,
    }))
}

fn is_executable(path: &Path) -> bool {
    fs::metadata(path)
        .map(|meta| meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

fn probe_helper_direct(helper: &Path) -> Value {
    match Command::new(helper).arg("probe").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let parsed: Value = serde_json::from_str(&stdout).unwrap_or_else(|_| json!({}));
            json!({
                "ok": output.status.success(),
                "status": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
                "json": parsed,
                "details": parsed.get("details").cloned().unwrap_or_else(|| json!({})),
            })
        }
        Err(err) => json!({
            "ok": false,
            "error": err.to_string(),
        }),
    }
}

fn privileged_fix_command(helper: Option<&Path>, pkexec: Option<&Path>, args: &[&str]) -> String {
    let helper = helper
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "hipfire-priv-helper".to_string());
    let runner = pkexec
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "sudo".to_string());
    std::iter::once(runner)
        .chain(std::iter::once(helper))
        .chain(args.iter().map(|arg| arg.to_string()))
        .map(|part| shell_quote(&part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-' | b':' | b'+')
        })
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn find_priv_helper() -> Option<PathBuf> {
    let exe_neighbor = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("hipfire-priv-helper")));
    exe_neighbor
        .filter(|path| path.is_file())
        .or_else(|| {
            let candidate = hipfire_dir().join("bin").join("hipfire-priv-helper");
            candidate.is_file().then_some(candidate)
        })
        .or_else(|| find_in_path("hipfire-priv-helper"))
}

fn is_mountpoint(path: &str) -> bool {
    fs::read_to_string("/proc/self/mountinfo")
        .map(|mountinfo| {
            mountinfo.lines().any(|line| {
                line.split_ascii_whitespace()
                    .nth(4)
                    .is_some_and(|mount| mount == path)
            })
        })
        .unwrap_or(false)
}

fn workspace_root() -> Option<PathBuf> {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|path| path.parent()?.parent().map(Path::to_path_buf))
        .or_else(|| std::env::current_dir().ok())
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1103_cwsr_off_reports_active_workaround() {
        let check = build_gfx1103_cwsr_check("gfx1103", Ok("0".to_string())).unwrap();

        assert_eq!(check.id, "driver.gpu_cwsr");
        assert_eq!(check.status, CheckStatus::Pass);
        assert_eq!(check.details["enabled"], json!(false));
        assert_eq!(check.details["workaround_active"], json!(true));
        assert!(check.fix.is_none());
    }

    #[test]
    fn gfx1103_cwsr_on_warns_with_reboot_remedy() {
        let check = build_gfx1103_cwsr_check("gfx1103", Ok("Y".to_string())).unwrap();

        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.details["enabled"], json!(true));
        assert_eq!(check.details["workaround_active"], json!(false));
        assert!(check
            .fix
            .as_deref()
            .is_some_and(|fix| fix.contains("amdgpu.cwsr_enable=0")));
    }

    #[test]
    fn gfx1103_unreadable_cwsr_warns() {
        let check =
            build_gfx1103_cwsr_check("gfx1103", Err("permission denied".to_string())).unwrap();

        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.details["error"], json!("permission denied"));
        assert_eq!(check.details["enabled"], Value::Null);
    }

    #[test]
    fn cwsr_check_is_gfx1103_only() {
        assert!(build_gfx1103_cwsr_check("gfx1151", Ok("1".to_string())).is_none());
        assert!(build_gfx1103_cwsr_check("gfx1100:sramecc+", Ok("0".to_string())).is_none());
    }

    #[test]
    fn cwsr_parser_accepts_kernel_boolean_spellings() {
        for value in ["0", "N", "no", "FALSE"] {
            assert_eq!(parse_module_bool(value), Some(false));
        }
        for value in ["1", "Y", "yes", "TRUE"] {
            assert_eq!(parse_module_bool(value), Some(true));
        }
        assert_eq!(parse_module_bool("unexpected"), None);
    }
}

use std::collections::BTreeMap;

use hipfire_admin_types::{
    AccessRateLimitRow, AccessUsageResponse, AccessUsageRow, AccessUser, CursorPage,
};
use leptos::prelude::*;

use crate::{AuthState, PageHead};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Window {
    Day,
    Week,
    Month,
    Quarter,
}

impl Window {
    fn seconds(self) -> u64 {
        match self {
            Self::Day => 86_400,
            Self::Week => 604_800,
            Self::Month => 2_592_000,
            Self::Quarter => 7_776_000,
        }
    }
}

#[component]
pub fn UsagePanel() -> impl IntoView {
    let auth = use_context::<AuthState>().expect("auth context");
    let (window, set_window) = signal(Window::Day);
    let (user_id, set_user_id) = signal(String::new());
    let (token_id, set_token_id) = signal(String::new());
    let (workload, set_workload) = signal(String::new());
    let (refresh, set_refresh) = signal(0_u32);
    let (usage, set_usage) = signal(None::<Result<AccessUsageResponse, String>>);
    let (rates, set_rates) = signal(None::<Result<Vec<AccessRateLimitRow>, String>>);
    let (users, set_users) = signal(Vec::<AccessUser>::new());

    leptos::task::spawn_local(async move {
        if let Ok(page) = hipfire_web_ui::get_json_typed::<CursorPage<AccessUser>>(
            "/admin/access/users?limit=200",
        )
        .await
        {
            set_users.set(page.items);
        }
    });

    Effect::new(move |_| {
        let _generation = refresh.get();
        let selected_window = window.get();
        let selected_user = user_id.get();
        let selected_token = token_id.get();
        let selected_workload = workload.get();
        set_usage.set(None);
        set_rates.set(None);
        leptos::task::spawn_local(async move {
            let now = (js_sys::Date::now() / 1000.0) as u64;
            let mut usage_url = format!(
                "/admin/access/usage?limit=1000&from={}",
                now.saturating_sub(selected_window.seconds())
            );
            let mut rate_url = "/admin/access/rate-limits?limit=1000".to_string();
            if !selected_user.is_empty() {
                usage_url.push_str(&format!("&user_id={selected_user}"));
                rate_url.push_str(&format!("&user_id={selected_user}"));
            }
            if !selected_token.is_empty() {
                usage_url.push_str(&format!("&token_id={selected_token}"));
                rate_url.push_str(&format!("&token_id={selected_token}"));
            }
            if !selected_workload.is_empty() {
                usage_url.push_str(&format!("&workload={selected_workload}"));
            }
            let usage_result =
                hipfire_web_ui::get_json_typed::<AccessUsageResponse>(&usage_url).await;
            let rate_result =
                hipfire_web_ui::get_json_typed::<CursorPage<AccessRateLimitRow>>(&rate_url).await;
            if usage_result
                .as_ref()
                .is_err_and(|error| error.is_unauthorized())
                || rate_result
                    .as_ref()
                    .is_err_and(|error| error.is_unauthorized())
            {
                auth.0.set(true);
            }
            set_usage.set(Some(usage_result.map_err(|error| error.to_string())));
            set_rates.set(Some(
                rate_result
                    .map(|page| page.items)
                    .map_err(|error| error.to_string()),
            ));
        });
    });

    view! {
        <PageHead eyebrow="Observability" title="Usage & limits" description="Privacy-safe hourly rollups and the live state of every rate bucket."/>
        <div class="toolbar usage-toolbar">
            <div class="period-tabs" role="group" aria-label="Usage period">
                <button class:active=move || window.get() == Window::Day on:click=move |_| set_window.set(Window::Day)>"24h"</button>
                <button class:active=move || window.get() == Window::Week on:click=move |_| set_window.set(Window::Week)>"7d"</button>
                <button class:active=move || window.get() == Window::Month on:click=move |_| set_window.set(Window::Month)>"30d"</button>
                <button class:active=move || window.get() == Window::Quarter on:click=move |_| set_window.set(Window::Quarter)>"90d"</button>
            </div>
            <button on:click=move |_| set_refresh.update(|value| *value += 1)>"Refresh"</button>
        </div>
        <div class="filter-bar">
            <label>"User"<select on:change=move |event| set_user_id.set(event_target_value(&event))><option value="">"All users"</option>{move || users.get().into_iter().map(|user| view! { <option value=user.id>{user.name}</option> }).collect_view()}</select></label>
            <label>"Token ID"<input placeholder="All tokens" on:input=move |event| set_token_id.set(event_target_value(&event))/></label>
            <label>"Workload"<select on:change=move |event| set_workload.set(event_target_value(&event))><option value="">"All workloads"</option><option value="text">"Text"</option><option value="embeddings">"Embeddings"</option><option value="images">"Images"</option><option value="training">"Training"</option></select></label>
        </div>
        {move || match usage.get() {
            None => view! { <UsageSkeleton/> }.into_any(),
            Some(Err(error)) => view! { <div class="error-panel" role="alert"><strong>"Usage unavailable"</strong><p>{error}</p></div> }.into_any(),
            Some(Ok(response)) => view! { <UsageContent response=response/> }.into_any(),
        }}
        <section class="panel rate-panel">
            <div class="panel-head"><div><h2>"Current rate state"</h2><p>"Remaining capacity and active concurrency at this instant"</p></div></div>
            {move || match rates.get() {
                None => view! { <div class="list-skeleton"><span></span><span></span></div> }.into_any(),
                Some(Err(error)) => view! { <div class="empty error"><p>{error}</p></div> }.into_any(),
                Some(Ok(rows)) if rows.is_empty() => view! { <div class="empty compact"><strong>"No matching rate buckets"</strong><p>"Create a user or clear the filters."</p></div> }.into_any(),
                Some(Ok(rows)) => view! { <RateTable rows=rows/> }.into_any(),
            }}
        </section>
    }
}

#[component]
fn UsageContent(response: AccessUsageResponse) -> impl IntoView {
    let totals = response.totals.clone();
    let rows = response.rows.items;
    view! {
        <div class="metric-grid">
            <MetricCard label="Requests" value=fmt_num(totals.requests) note=format!("{} errors", fmt_num(totals.errors))/>
            <MetricCard label="Input tokens" value=fmt_num(totals.input_tokens) note=format!("{} cached", fmt_num(totals.cache_tokens))/>
            <MetricCard label="Output tokens" value=fmt_num(totals.output_tokens) note="generated".to_string()/>
            <MetricCard label="Rate-limit hits" value=fmt_num(totals.rate_limit_hits) note="HTTP 429 responses".to_string()/>
        </div>
        <section class="panel chart-panel">
            <div class="panel-head"><div><h2>"Request trend"</h2><p>"Hourly aggregate; request content is never retained"</p></div></div>
            <UsageChart rows=rows.clone()/>
        </section>
        <section class="panel breakdown-panel">
            <div class="panel-head"><div><h2>"Usage breakdown"</h2><p>"Hourly rows by user, token, and workload"</p></div></div>
            {if rows.is_empty() { view! { <div class="empty compact"><strong>"No usage in this period"</strong><p>"Traffic will appear here after the first accounted request."</p></div> }.into_any() } else { view! { <UsageTable rows=rows/> }.into_any() }}
        </section>
    }
}

#[component]
fn MetricCard(label: &'static str, value: String, note: String) -> impl IntoView {
    view! { <div class="metric-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div> }
}

#[component]
fn UsageChart(rows: Vec<AccessUsageRow>) -> impl IntoView {
    let mut hours = BTreeMap::<u64, u64>::new();
    for row in rows {
        *hours.entry(row.hour_start).or_default() += row.counters.requests;
    }
    let max = hours.values().copied().max().unwrap_or(1).max(1);
    if hours.is_empty() {
        return view! { <div class="empty compact"><p>"No request activity to chart."</p></div> }
            .into_any();
    }
    view! { <div class="usage-chart" role="img" aria-label="Hourly request volume">{hours.into_iter().map(|(hour, count)| { let height = (count as f64 / max as f64 * 100.0).max(3.0); view! { <div class="chart-column" title=format!("{} requests at unix hour {}", count, hour)><span style=format!("height:{height:.1}%")></span></div> } }).collect_view()}</div> }.into_any()
}

#[component]
fn UsageTable(rows: Vec<AccessUsageRow>) -> impl IntoView {
    view! { <div class="table-scroll"><table><thead><tr><th>"Hour (Unix)"</th><th>"User"</th><th>"Token"</th><th>"Workload"</th><th class="num">"Requests"</th><th class="num">"Tokens in / out"</th><th class="num">"429s"</th></tr></thead><tbody>{rows.into_iter().map(|row| view! { <tr><td class="mono">{row.hour_start}</td><td class="mono">{short_id(&row.user_id)}</td><td class="mono">{short_id(&row.token_id)}</td><td>{row.workload}</td><td class="num">{fmt_num(row.counters.requests)}</td><td class="num">{format!("{} / {}", fmt_num(row.counters.input_tokens), fmt_num(row.counters.output_tokens))}</td><td class="num">{fmt_num(row.counters.rate_limit_hits)}</td></tr> }).collect_view()}</tbody></table></div> }
}

#[component]
fn RateTable(rows: Vec<AccessRateLimitRow>) -> impl IntoView {
    view! { <div class="table-scroll"><table><thead><tr><th>"User"</th><th>"Token"</th><th class="num">"Requests left"</th><th class="num">"Text tokens left"</th><th class="num">"Active text"</th><th class="num">"Images"</th><th class="num">"Training"</th></tr></thead><tbody>{rows.into_iter().map(|row| view! { <tr><td class="mono">{short_id(&row.user_id)}</td><td class="mono">{row.token_id.as_deref().map(short_id).unwrap_or_else(|| "user aggregate".into())}</td><td class="num">{format!("{:.0}", row.request_remaining)}</td><td class="num">{format!("{:.0}", row.text_token_remaining)}</td><td class="num">{format!("{} / {}", row.active_text, row.effective_policy.max_in_flight_text)}</td><td class="num">{format!("{} / {}", row.active_images, row.effective_policy.max_in_flight_images)}</td><td class="num">{format!("{} / {}", row.active_training, row.effective_policy.max_in_flight_training)}</td></tr> }).collect_view()}</tbody></table></div> }
}

#[component]
fn UsageSkeleton() -> impl IntoView {
    view! { <><div class="metric-grid"><div class="skeleton metric"></div><div class="skeleton metric"></div><div class="skeleton metric"></div><div class="skeleton metric"></div></div><div class="skeleton chart"></div></> }
}

fn fmt_num(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}
fn short_id(id: &str) -> String {
    if id.len() > 14 {
        format!("{}…{}", &id[..8], &id[id.len() - 4..])
    } else {
        id.to_string()
    }
}

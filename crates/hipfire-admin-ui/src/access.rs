use hipfire_admin_types::{
    AccessRatePolicy, AccessScope, AccessToken, AccessUser, AccessUserStatus,
    CreateAccessTokenRequest, CreateAccessUserRequest, CreatedAccessToken, CursorPage,
    PatchAccessUserRequest,
};
use leptos::prelude::*;

use crate::{AuthState, PageHead};

#[component]
pub fn AccessPanel() -> impl IntoView {
    let auth = use_context::<AuthState>().expect("auth context");
    let (users, set_users) = signal(None::<Result<Vec<AccessUser>, String>>);
    let (selected_id, set_selected_id) = signal(None::<String>);
    let (detail, set_detail) = signal(None::<Result<AccessUser, String>>);
    let (tokens, set_tokens) = signal(Vec::<AccessToken>::new());
    let (search, set_search) = signal(String::new());
    let (status, set_status) = signal("all".to_string());
    let (create_open, set_create_open) = signal(false);
    let (new_name, set_new_name) = signal(String::new());
    let (notice, set_notice) = signal(None::<String>);

    let reload_users = move || {
        set_users.set(None);
        leptos::task::spawn_local(async move {
            match hipfire_web_ui::get_json_typed::<CursorPage<AccessUser>>(
                "/admin/access/users?limit=200",
            )
            .await
            {
                Ok(page) => set_users.set(Some(Ok(page.items))),
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_users.set(Some(Err(error.to_string())));
                }
            }
        });
    };
    reload_users();

    Effect::new(move |_| {
        let Some(id) = selected_id.get() else {
            set_detail.set(None);
            set_tokens.set(Vec::new());
            return;
        };
        set_detail.set(None);
        leptos::task::spawn_local(async move {
            let user =
                hipfire_web_ui::get_json_typed::<AccessUser>(&format!("/admin/access/users/{id}"))
                    .await;
            let token_page = hipfire_web_ui::get_json_typed::<CursorPage<AccessToken>>(&format!(
                "/admin/access/users/{id}/tokens?limit=200"
            ))
            .await;
            match user {
                Ok(user) => set_detail.set(Some(Ok(user))),
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_detail.set(Some(Err(error.to_string())));
                }
            }
            match token_page {
                Ok(page) => set_tokens.set(page.items),
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                }
            }
        });
    });

    let create_user = move || {
        let name = new_name.get_untracked().trim().to_string();
        if name.is_empty() {
            return;
        }
        leptos::task::spawn_local(async move {
            let body = CreateAccessUserRequest {
                name,
                rate_policy: AccessRatePolicy::default(),
            };
            match hipfire_web_ui::post_json_typed::<_, AccessUser>("/admin/access/users", &body)
                .await
            {
                Ok(user) => {
                    set_notice.set(Some(format!("Created {}.", user.name)));
                    set_new_name.set(String::new());
                    set_create_open.set(false);
                    set_selected_id.set(Some(user.id));
                    reload_users();
                }
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_notice.set(Some(error.to_string()));
                }
            }
        });
    };

    view! {
        <PageHead eyebrow="Credentials" title="API access" description="Create users, issue scoped tokens, and enforce workload limits."/>
        {move || notice.get().map(|message| view! { <div class="notice" role="status">{message}</div> })}
        <div class="toolbar">
            <div class="filter-group">
                <label class="search-field"><span class="sr-only">"Search users"</span><input type="search" placeholder="Search users…" on:input=move |event| set_search.set(event_target_value(&event))/></label>
                <label><span class="sr-only">"Filter by status"</span><select on:change=move |event| set_status.set(event_target_value(&event))><option value="all">"All statuses"</option><option value="enabled">"Enabled"</option><option value="disabled">"Disabled"</option></select></label>
            </div>
            <button class="primary" on:click=move |_| set_create_open.set(true)>"New API user"</button>
        </div>
        {move || create_open.get().then(|| view! {
            <form class="inline-create" on:submit=move |event: leptos::ev::SubmitEvent| { event.prevent_default(); create_user(); }>
                <div><strong>"Create API user"</strong><p>"A user owns tokens, limits, usage, and response context."</p></div>
                <label>"Display name"<input autofocus prop:value=move || new_name.get() on:input=move |event| set_new_name.set(event_target_value(&event))/></label>
                <div class="actions"><button type="button" on:click=move |_| set_create_open.set(false)>"Cancel"</button><button class="primary" type="submit">"Create user"</button></div>
            </form>
        })}
        <div class="access-layout">
            <section class="panel user-list" aria-labelledby="users-title">
                <div class="panel-head"><div><h2 id="users-title">"Users"</h2><p>"API identities on this server"</p></div></div>
                {move || match users.get() {
                    None => view! { <ListSkeleton/> }.into_any(),
                    Some(Err(error)) => view! { <div class="empty error"><strong>"Users unavailable"</strong><p>{error}</p><button on:click=move |_| reload_users()>"Try again"</button></div> }.into_any(),
                    Some(Ok(items)) => {
                        let needle = search.get().to_lowercase();
                        let wanted = status.get();
                        let visible = items.into_iter().filter(|user| {
                            (needle.is_empty() || user.name.to_lowercase().contains(&needle)) &&
                            (wanted == "all" || wanted == status_name(user.status))
                        }).collect::<Vec<_>>();
                        if visible.is_empty() {
                            view! { <div class="empty"><strong>"No matching users"</strong><p>"Adjust the filters or create the first API user."</p></div> }.into_any()
                        } else {
                            visible.into_iter().map(|user| {
                                let id = user.id.clone();
                                let active = id.clone();
                                view! {
                                    <button class="user-row" class:selected=move || selected_id.get().as_deref() == Some(active.as_str()) on:click=move |_| set_selected_id.set(Some(id.clone()))>
                                        <span><strong>{user.name}</strong><small class="mono">{short_id(&user.id)}</small></span>
                                        <span class=format!("status {}", status_name(user.status))>{status_name(user.status)}</span>
                                        <small>{format!("{} token{}", user.token_count, if user.token_count == 1 { "" } else { "s" })}</small>
                                    </button>
                                }
                            }).collect_view().into_any()
                        }
                    }
                }}
            </section>
            <section class="panel detail-panel" aria-live="polite">
                {move || match detail.get() {
                    None if selected_id.get().is_none() => view! { <div class="empty generous"><span class="empty-icon">"↳"</span><strong>"Select an API user"</strong><p>"Inspect credentials and tune limits without exposing token secrets."</p></div> }.into_any(),
                    None => view! { <ListSkeleton/> }.into_any(),
                    Some(Err(error)) => view! { <div class="empty error"><strong>"User unavailable"</strong><p>{error}</p></div> }.into_any(),
                    Some(Ok(user)) => view! { <UserDetail user=user set_detail=set_detail tokens=tokens set_tokens=set_tokens reload_users=Callback::new(move |_| reload_users())/> }.into_any(),
                }}
            </section>
        </div>
    }
}

#[component]
fn UserDetail(
    user: AccessUser,
    set_detail: WriteSignal<Option<Result<AccessUser, String>>>,
    tokens: ReadSignal<Vec<AccessToken>>,
    set_tokens: WriteSignal<Vec<AccessToken>>,
    reload_users: Callback<()>,
) -> impl IntoView {
    let auth = use_context::<AuthState>().expect("auth context");
    let user_id = user.id.clone();
    let (editing, set_editing) = signal(user.clone());
    let (confirm_disable, set_confirm_disable) = signal(false);
    let (token_label, set_token_label) = signal(String::new());
    let (scope_text, set_scope_text) = signal(true);
    let (scope_embeddings, set_scope_embeddings) = signal(false);
    let (scope_images, set_scope_images) = signal(false);
    let (scope_training, set_scope_training) = signal(false);
    let (created, set_created) = signal(None::<CreatedAccessToken>);
    let (copied, set_copied) = signal(false);
    let (confirm_revoke, set_confirm_revoke) = signal(None::<String>);
    let (feedback, set_feedback) = signal(None::<String>);

    let limits_user_id = user_id.clone();
    let save_limits = move || {
        let body = PatchAccessUserRequest {
            status: None,
            rate_policy: Some(editing.get_untracked().rate_policy),
        };
        let id = limits_user_id.clone();
        leptos::task::spawn_local(async move {
            match hipfire_web_ui::patch_json::<_, AccessUser>(
                &format!("/admin/access/users/{id}"),
                &body,
            )
            .await
            {
                Ok(user) => {
                    set_detail.set(Some(Ok(user.clone())));
                    set_editing.set(user);
                    set_feedback.set(Some("Limits saved.".into()));
                    reload_users.run(());
                }
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_feedback.set(Some(error.to_string()));
                }
            }
        });
    };
    let status_user_id = user_id.clone();
    let toggle_status = Callback::new(move |_: ()| {
        let current = editing.get_untracked().status;
        let next = if current == AccessUserStatus::Enabled {
            AccessUserStatus::Disabled
        } else {
            AccessUserStatus::Enabled
        };
        let body = PatchAccessUserRequest {
            status: Some(next),
            rate_policy: None,
        };
        let id = status_user_id.clone();
        leptos::task::spawn_local(async move {
            match hipfire_web_ui::patch_json::<_, AccessUser>(
                &format!("/admin/access/users/{id}"),
                &body,
            )
            .await
            {
                Ok(user) => {
                    set_detail.set(Some(Ok(user.clone())));
                    set_editing.set(user);
                    set_confirm_disable.set(false);
                    reload_users.run(());
                }
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_feedback.set(Some(error.to_string()));
                }
            }
        });
    });
    let token_user_id = user_id.clone();
    let issue_token = move || {
        let mut scopes = Vec::new();
        if scope_text.get_untracked() {
            scopes.push(AccessScope::Text);
        }
        if scope_embeddings.get_untracked() {
            scopes.push(AccessScope::Embeddings);
        }
        if scope_images.get_untracked() {
            scopes.push(AccessScope::Images);
        }
        if scope_training.get_untracked() {
            scopes.push(AccessScope::Training);
        }
        let label = token_label.get_untracked().trim().to_string();
        if label.is_empty() || scopes.is_empty() {
            set_feedback.set(Some("Add a label and at least one scope.".into()));
            return;
        }
        let body = CreateAccessTokenRequest {
            label,
            scopes,
            rate_policy: AccessRatePolicy::default(),
            expires_at: None,
        };
        let id = token_user_id.clone();
        leptos::task::spawn_local(async move {
            match hipfire_web_ui::post_json_typed::<_, CreatedAccessToken>(
                &format!("/admin/access/users/{id}/tokens"),
                &body,
            )
            .await
            {
                Ok(value) => {
                    set_tokens.update(|items| items.push(value.token.clone()));
                    set_created.set(Some(value));
                    set_token_label.set(String::new());
                }
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_feedback.set(Some(error.to_string()));
                }
            }
        });
    };
    let revoke = move |id: String| {
        let request_id = id.clone();
        leptos::task::spawn_local(async move {
            match hipfire_web_ui::delete_json::<serde_json::Value>(&format!(
                "/admin/access/tokens/{request_id}"
            ))
            .await
            {
                Ok(_) => {
                    set_tokens.update(|items| {
                        if let Some(token) = items.iter_mut().find(|token| token.id == id) {
                            token.revoked_at = Some(1);
                        }
                    });
                    set_confirm_revoke.set(None);
                    set_feedback.set(Some("Token revoked.".into()));
                }
                Err(error) => {
                    if error.is_unauthorized() {
                        auth.0.set(true);
                    }
                    set_feedback.set(Some(error.to_string()));
                }
            }
        });
    };

    view! {
        <div class="detail-head">
            <div><p class="eyebrow">"API user"</p><h2>{user.name}</h2><p class="mono">{user.id.clone()}</p></div>
            <span class=format!("status {}", status_name(user.status))>{status_name(user.status)}</span>
        </div>
        {move || feedback.get().map(|message| view! { <p class="inline-feedback" role="status">{message}</p> })}
        <section class="detail-section">
            <div class="section-title"><div><h3>"Workload limits"</h3><p>"Blank fields inherit server defaults. Token overrides can only be stricter."</p></div><button class="primary compact" on:click=move |_| save_limits()>"Save limits"</button></div>
            <div class="limit-grid">
                <LimitInput label="Requests / minute" field=PolicyField::Requests policy=editing set_policy=set_editing/>
                <LimitInput label="Request burst" field=PolicyField::RequestBurst policy=editing set_policy=set_editing/>
                <LimitInput label="Text tokens / minute" field=PolicyField::TextTokens policy=editing set_policy=set_editing/>
                <LimitInput label="Text token burst" field=PolicyField::TextBurst policy=editing set_policy=set_editing/>
                <LimitInput label="Concurrent text" field=PolicyField::TextConcurrency policy=editing set_policy=set_editing/>
                <LimitInput label="Concurrent images" field=PolicyField::ImageConcurrency policy=editing set_policy=set_editing/>
                <LimitInput label="MP-steps / minute" field=PolicyField::MegapixelSteps policy=editing set_policy=set_editing/>
                <LimitInput label="MP-step burst" field=PolicyField::MegapixelBurst policy=editing set_policy=set_editing/>
                <LimitInput label="Concurrent training" field=PolicyField::TrainingConcurrency policy=editing set_policy=set_editing/>
            </div>
        </section>
        <section class="detail-section">
            <div class="section-title"><div><h3>"Tokens"</h3><p>"Secrets are shown once and never stored in recoverable form."</p></div></div>
            {move || created.get().map(|value| {
                let secret = value.secret.clone();
                view! { <div class="secret-panel" role="alert"><div><strong>"Copy this token now"</strong><p>"It cannot be shown again after this panel is closed."</p></div><code>{secret.clone()}</code><div class="actions"><button on:click=move |_| { if let Some(window) = web_sys::window() { let _ = window.navigator().clipboard().write_text(&secret); set_copied.set(true); } }>{move || if copied.get() { "Copied" } else { "Copy token" }}</button><button class="primary" on:click=move |_| set_created.set(None)>"I saved it"</button></div></div> }
            })}
            <form class="token-create" on:submit=move |event: leptos::ev::SubmitEvent| { event.prevent_default(); issue_token(); }>
                <label>"Token label"<input placeholder="production-client" prop:value=move || token_label.get() on:input=move |event| set_token_label.set(event_target_value(&event))/></label>
                <fieldset><legend>"Scopes"</legend><label><input type="checkbox" prop:checked=move || scope_text.get() on:change=move |event| set_scope_text.set(event_target_checked(&event))/>"Text"</label><label><input type="checkbox" prop:checked=move || scope_embeddings.get() on:change=move |event| set_scope_embeddings.set(event_target_checked(&event))/>"Embeddings"</label><label><input type="checkbox" prop:checked=move || scope_images.get() on:change=move |event| set_scope_images.set(event_target_checked(&event))/>"Images"</label><label><input type="checkbox" prop:checked=move || scope_training.get() on:change=move |event| set_scope_training.set(event_target_checked(&event))/>"Training"</label></fieldset>
                <button class="primary" type="submit">"Generate token"</button>
            </form>
            <div class="token-list">
                {move || if tokens.get().is_empty() { view! { <div class="empty compact"><strong>"No tokens yet"</strong><p>"Generate a scoped credential for this user."</p></div> }.into_any() } else { tokens.get().into_iter().map(|token| {
                    let id = token.id.clone(); let confirm_id = id.clone(); let revoke_id = id.clone();
                    let scopes = token.scopes.iter().map(scope_name).collect::<Vec<_>>().join(", ");
                    view! { <div class="token-row"><span><strong>{token.label}</strong><small class="mono">{short_id(&token.id)}</small></span><span class="scope-list">{scopes}</span>{if token.revoked_at.is_some() { view! { <span class="status disabled">"revoked"</span> }.into_any() } else if confirm_revoke.get().as_deref() == Some(confirm_id.as_str()) { view! { <span class="confirm-inline">"Revoke?" <button on:click=move |_| set_confirm_revoke.set(None)>"Cancel"</button><button class="danger" on:click=move |_| revoke(revoke_id.clone())>"Confirm"</button></span> }.into_any() } else { view! { <button class="danger quiet" on:click=move |_| set_confirm_revoke.set(Some(id.clone()))>"Revoke"</button> }.into_any() }}</div> }
                }).collect_view().into_any() }}
            </div>
        </section>
        <section class="detail-section danger-zone">
            <div><h3>{move || if editing.get().status == AccessUserStatus::Enabled { "Disable user" } else { "Enable user" }}</h3><p>{move || if editing.get().status == AccessUserStatus::Enabled { "New requests fail immediately; queued work is cancelled, active work may finish." } else { "Allow valid, unexpired tokens to authenticate again." }}</p></div>
            {move || if confirm_disable.get() { view! { <div class="confirm-inline"><button on:click=move |_| set_confirm_disable.set(false)>"Cancel"</button><button class="danger" on:click=move |_| toggle_status.run(())>"Confirm status change"</button></div> }.into_any() } else { view! { <button class="danger quiet" on:click=move |_| set_confirm_disable.set(true)>{if editing.get().status == AccessUserStatus::Enabled { "Disable…" } else { "Enable…" }}</button> }.into_any() }}
        </section>
    }
}

#[derive(Clone, Copy)]
enum PolicyField {
    Requests,
    RequestBurst,
    TextTokens,
    TextBurst,
    TextConcurrency,
    ImageConcurrency,
    MegapixelSteps,
    MegapixelBurst,
    TrainingConcurrency,
}

#[component]
fn LimitInput(
    label: &'static str,
    field: PolicyField,
    policy: ReadSignal<AccessUser>,
    set_policy: WriteSignal<AccessUser>,
) -> impl IntoView {
    let value = move || {
        let p = policy.get().rate_policy;
        match field {
            PolicyField::Requests => p.requests_per_minute.map(|v| v.to_string()),
            PolicyField::RequestBurst => p.request_burst.map(|v| v.to_string()),
            PolicyField::TextTokens => p.text_tokens_per_minute.map(|v| v.to_string()),
            PolicyField::TextBurst => p.text_token_burst.map(|v| v.to_string()),
            PolicyField::TextConcurrency => p.max_in_flight_text.map(|v| v.to_string()),
            PolicyField::ImageConcurrency => p.max_in_flight_images.map(|v| v.to_string()),
            PolicyField::MegapixelSteps => p.megapixel_steps_per_minute.map(|v| v.to_string()),
            PolicyField::MegapixelBurst => p.megapixel_step_burst.map(|v| v.to_string()),
            PolicyField::TrainingConcurrency => p.max_in_flight_training.map(|v| v.to_string()),
        }
        .unwrap_or_default()
    };
    view! { <label>{label}<input type="number" min="0" placeholder="Default" prop:value=value on:input=move |event| { let raw = event_target_value(&event); set_policy.update(|user| set_policy_field(&mut user.rate_policy, field, &raw)); }/></label> }
}

fn set_policy_field(policy: &mut AccessRatePolicy, field: PolicyField, raw: &str) {
    let wide = raw.parse::<u64>().ok();
    let narrow = raw.parse::<u32>().ok();
    match field {
        PolicyField::Requests => policy.requests_per_minute = wide,
        PolicyField::RequestBurst => policy.request_burst = wide,
        PolicyField::TextTokens => policy.text_tokens_per_minute = wide,
        PolicyField::TextBurst => policy.text_token_burst = wide,
        PolicyField::TextConcurrency => policy.max_in_flight_text = narrow,
        PolicyField::ImageConcurrency => policy.max_in_flight_images = narrow,
        PolicyField::MegapixelSteps => policy.megapixel_steps_per_minute = wide,
        PolicyField::MegapixelBurst => policy.megapixel_step_burst = wide,
        PolicyField::TrainingConcurrency => policy.max_in_flight_training = narrow,
    }
}

#[component]
fn ListSkeleton() -> impl IntoView {
    view! { <div class="list-skeleton"><span></span><span></span><span></span></div> }
}
fn status_name(status: AccessUserStatus) -> &'static str {
    match status {
        AccessUserStatus::Enabled => "enabled",
        AccessUserStatus::Disabled => "disabled",
    }
}
fn scope_name(scope: &AccessScope) -> &'static str {
    match scope {
        AccessScope::Text => "text",
        AccessScope::Embeddings => "embeddings",
        AccessScope::Images => "images",
        AccessScope::Training => "training",
    }
}
fn short_id(id: &str) -> String {
    if id.len() > 14 {
        format!("{}…{}", &id[..8], &id[id.len() - 4..])
    } else {
        id.to_string()
    }
}

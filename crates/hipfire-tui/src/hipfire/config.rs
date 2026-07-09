// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{collections::BTreeMap, fs, time::Duration};

use hipfire_config::{
    apply_config_edit, build_config_editor_snapshot_from_paths, config_schema,
    loaded_config_from_documents, ConfigEditOperation, ConfigEditRequest, ConfigEditTarget,
    ConfigEditorPaths, ConfigEditorSnapshot,
};
use serde_json::{Map, Value};

use super::HipfirePaths;

const EASY_KEYS: &[(&str, &str)] = &[
    ("default_model", "Model"),
    ("max_seq", "Context"),
    ("dflash_mode", "Spec decode"),
    ("kv_cache", "KV cache"),
    ("thinking", "Thinking"),
];

#[derive(Clone, Debug)]
pub struct ConfigState {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub per_model_count: usize,
    pub warning: Option<String>,
    pub schema_field_count: Option<usize>,
    pub schema_warning: Option<String>,
    pub resolved_from_daemon: bool,
    easy_rows: Vec<EasyConfigRow>,
    advanced_rows: Vec<EasyConfigRow>,
}

#[derive(Clone, Copy, Debug)]
pub enum ConfigEditDirection {
    Previous,
    Next,
}

impl ConfigState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let mut warning = None;
        let mut values = defaults();
        let local_documents = read_local_documents(paths, &mut warning);
        for (key, value) in values_from_document(&local_documents.global) {
            values.insert(key, value);
        }
        for (key, value) in values_from_document(&local_documents.host) {
            values.insert(key, value);
        }

        let probe_host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "127.0.0.1".into());
        let probe_port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);

        let per_model_count = fs::read_to_string(&paths.per_model_config)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().map(|m| m.len()))
            .unwrap_or(0);

        let editor = match load_remote_editor(&probe_host_for(&probe_host), probe_port) {
            Ok(editor) => editor,
            Err(err) => {
                let snapshot = local_editor_snapshot(paths, local_documents);
                EditorRows::from_snapshot(snapshot, false, Some(err))
            }
        };

        let mut values = editor.values;
        if values.is_empty() {
            values = defaults();
        }
        let host = values
            .get("host")
            .cloned()
            .unwrap_or_else(|| "127.0.0.1".into());
        let port = values
            .get("port")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11435);
        let default_model = values
            .get("default_model")
            .filter(|model| !model.is_empty())
            .cloned()
            .unwrap_or_else(|| "unset".into());

        Self {
            host,
            port,
            default_model,
            per_model_count,
            warning,
            schema_field_count: Some(editor.advanced_rows.len()),
            schema_warning: editor.warning,
            resolved_from_daemon: editor.remote,
            easy_rows: editor.easy_rows,
            advanced_rows: editor.advanced_rows,
        }
    }

    pub fn probe_host(&self) -> String {
        probe_host_for(&self.host)
    }

    pub fn easy_rows(&self) -> Vec<(String, String, String)> {
        let mut rows = self
            .easy_rows
            .iter()
            .map(|row| {
                let mut desc = row.description.clone();
                if row.pending {
                    desc = format!("{}; pending {}", desc, row.impact);
                } else if !row.impact.is_empty() {
                    desc = format!("{}; {}", desc, row.impact);
                }
                (row.label.clone(), row.value.clone(), desc)
            })
            .collect::<Vec<_>>();
        rows.push((
            "Serve".into(),
            format!("{}:{}", self.host, self.port),
            "OpenAI-compatible endpoint used by chat and API clients.".into(),
        ));
        rows.push((
            "Schema".into(),
            self.schema_field_count
                .map(|count| format!("{count} fields"))
                .unwrap_or_else(|| "offline".into()),
            self.schema_warning.clone().unwrap_or_else(|| {
                if self.resolved_from_daemon {
                    "Loaded active config from daemon editor API.".into()
                } else {
                    "Loaded local config editor snapshot.".into()
                }
            }),
        ));
        rows
    }

    pub fn advanced_rows(&self) -> &[EasyConfigRow] {
        &self.advanced_rows
    }

    pub fn edit_easy_row(
        &self,
        paths: &HipfirePaths,
        selected: usize,
        active_model: &str,
        direction: ConfigEditDirection,
    ) -> Result<String, String> {
        let Some(row) = self.easy_rows.get(selected) else {
            return Err("no setting selected".to_string());
        };
        self.edit_row(paths, row, active_model, direction)
    }

    pub fn edit_advanced_row(
        &self,
        paths: &HipfirePaths,
        selected: usize,
        direction: ConfigEditDirection,
    ) -> Result<String, String> {
        let Some(row) = self.advanced_rows.get(selected) else {
            return Err("no setting selected".to_string());
        };
        self.edit_row(paths, row, "", direction)
    }

    pub fn unset_easy_row(&self, paths: &HipfirePaths, selected: usize) -> Result<String, String> {
        let Some(row) = self.easy_rows.get(selected) else {
            return Err("no setting selected".to_string());
        };
        self.unset_row(paths, row)
    }

    pub fn unset_advanced_row(
        &self,
        paths: &HipfirePaths,
        selected: usize,
    ) -> Result<String, String> {
        let Some(row) = self.advanced_rows.get(selected) else {
            return Err("no setting selected".to_string());
        };
        self.unset_row(paths, row)
    }

    fn edit_row(
        &self,
        paths: &HipfirePaths,
        row: &EasyConfigRow,
        active_model: &str,
        direction: ConfigEditDirection,
    ) -> Result<String, String> {
        let Some(key) = row.key.as_deref() else {
            return Err(format!("{} is read-only", row.label));
        };
        if !row.editable {
            return Err(format!("{} is not editable in the global layer", row.label));
        }

        let next = if key == "default_model" {
            let active = active_model.trim();
            if active.is_empty() || active == "unset" {
                return Err("select a model in the Models tab first".to_string());
            }
            Value::String(active.to_string())
        } else if let Some(value) = cycle_row_value(row, direction) {
            value
        } else {
            return Err(format!("{} does not support cycling", row.label));
        };

        let active = local_loaded_config(paths);
        let paths = ConfigEditorPaths {
            global: paths.config.clone(),
            host: paths.host_config.clone(),
        };
        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Global,
                key: key.to_string(),
                operation: ConfigEditOperation::Set,
                value: Some(next.clone()),
                model: None,
            },
            &active,
        )?;
        Ok(format!(
            "saved {key}={} to {}; {}",
            value_to_string(&next),
            paths.global.display(),
            row.impact
        ))
    }

    fn unset_row(&self, paths: &HipfirePaths, row: &EasyConfigRow) -> Result<String, String> {
        let Some(key) = row.key.as_deref() else {
            return Err(format!("{} is read-only", row.label));
        };
        if !row.editable {
            return Err(format!("{} is not editable in the global layer", row.label));
        }
        let active = local_loaded_config(paths);
        let paths = ConfigEditorPaths {
            global: paths.config.clone(),
            host: paths.host_config.clone(),
        };
        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Global,
                key: key.to_string(),
                operation: ConfigEditOperation::Unset,
                value: None,
                model: None,
            },
            &active,
        )?;
        Ok(format!("unset {key} in {}", paths.global.display()))
    }
}

#[derive(Clone, Debug)]
pub struct EasyConfigRow {
    pub label: String,
    pub key: Option<String>,
    pub value: String,
    pub active_value: String,
    pub description: String,
    pub choices: Vec<String>,
    pub kind: String,
    pub impact: String,
    pub pending: bool,
    pub editable: bool,
}

#[derive(Clone, Debug)]
struct EditorRows {
    values: BTreeMap<String, String>,
    easy_rows: Vec<EasyConfigRow>,
    advanced_rows: Vec<EasyConfigRow>,
    remote: bool,
    warning: Option<String>,
}

impl EditorRows {
    fn from_snapshot(
        snapshot: ConfigEditorSnapshot,
        remote: bool,
        warning: Option<String>,
    ) -> Self {
        let values = snapshot
            .rows
            .iter()
            .filter_map(|row| {
                row.active_value
                    .as_ref()
                    .or(row.local_value.as_ref())
                    .map(|value| (row.key.clone(), value_to_string(value)))
            })
            .collect::<BTreeMap<_, _>>();
        let mut advanced_rows = snapshot
            .rows
            .iter()
            .map(|row| EasyConfigRow {
                label: row.key.clone(),
                key: Some(row.key.clone()),
                value: row
                    .local_value
                    .as_ref()
                    .map(value_to_string)
                    .unwrap_or_else(|| "unset".into()),
                active_value: row
                    .active_value
                    .as_ref()
                    .map(value_to_string)
                    .unwrap_or_else(|| "unset".into()),
                description: row.description.clone(),
                choices: row.enum_choices.clone(),
                kind: config_type_kind(&row.ty),
                impact: impact_label(row.mutability, row.restart_impact, row.pending),
                pending: row.pending,
                editable: row.editable_global,
            })
            .collect::<Vec<_>>();
        let easy_rows = easy_rows_from_advanced(&advanced_rows);
        advanced_rows.sort_by(|a, b| a.label.cmp(&b.label));
        Self {
            values,
            easy_rows,
            advanced_rows,
            remote,
            warning,
        }
    }

    fn from_remote(payload: Value) -> Result<Self, String> {
        let rows = payload
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| "config editor endpoint returned unexpected JSON".to_string())?;
        let mut values = BTreeMap::new();
        let mut advanced_rows = Vec::new();
        for row in rows {
            let key = row
                .get("key")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let local = row.get("local_value").unwrap_or(&Value::Null);
            let active = row.get("active_value").unwrap_or(local);
            values.insert(key.clone(), value_to_string(active));
            advanced_rows.push(EasyConfigRow {
                label: key.clone(),
                key: Some(key),
                value: value_to_string(local),
                active_value: value_to_string(active),
                description: row
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                choices: row
                    .get("enum_choices")
                    .and_then(Value::as_array)
                    .map(|choices| {
                        choices
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                kind: row
                    .get("type")
                    .and_then(|ty| ty.get("kind"))
                    .and_then(Value::as_str)
                    .unwrap_or("json")
                    .to_string(),
                impact: impact_from_json(row),
                pending: row.get("pending").and_then(Value::as_bool).unwrap_or(false),
                editable: row
                    .get("editable_global")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
        }
        let easy_rows = easy_rows_from_advanced(&advanced_rows);
        advanced_rows.sort_by(|a, b| a.label.cmp(&b.label));
        Ok(Self {
            values,
            easy_rows,
            advanced_rows,
            remote: true,
            warning: None,
        })
    }
}

struct LocalDocuments {
    global: Value,
    global_error: Option<String>,
    host: Value,
    host_error: Option<String>,
}

fn read_local_documents(paths: &HipfirePaths, warning: &mut Option<String>) -> LocalDocuments {
    let (global, global_error) = read_document(&paths.config);
    if let Some(err) = &global_error {
        *warning = Some(format!("config.json {err}"));
    }
    let (host, host_error) = read_document(&paths.host_config);
    LocalDocuments {
        global,
        global_error,
        host,
        host_error,
    }
}

fn local_editor_snapshot(paths: &HipfirePaths, documents: LocalDocuments) -> ConfigEditorSnapshot {
    let loaded = loaded_config_from_documents(
        paths.config.clone(),
        documents.global,
        documents.global_error,
        paths.host_config.clone(),
        documents.host,
        documents.host_error,
        Vec::new(),
    );
    let editor_paths = ConfigEditorPaths {
        global: paths.config.clone(),
        host: paths.host_config.clone(),
    };
    build_config_editor_snapshot_from_paths(&editor_paths, &loaded, None)
}

fn local_loaded_config(paths: &HipfirePaths) -> hipfire_config::LoadedConfig {
    let (global, global_error) = read_document(&paths.config);
    let (host, host_error) = read_document(&paths.host_config);
    loaded_config_from_documents(
        paths.config.clone(),
        global,
        global_error,
        paths.host_config.clone(),
        host,
        host_error,
        Vec::new(),
    )
}

fn load_remote_editor(host: &str, port: u16) -> Result<EditorRows, String> {
    let url = format!("http://{host}:{port}/admin/config/editor");
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(650))
        .build();
    let body = super::authorize_admin(agent.get(&url))
        .call()
        .map_err(|err| format!("editor config unavailable: {err}"))?
        .into_string()
        .map_err(|err| format!("editor config read error: {err}"))?;
    let payload = serde_json::from_str::<Value>(&body)
        .map_err(|err| format!("editor config parse error: {err}"))?;
    EditorRows::from_remote(payload)
}

fn easy_rows_from_advanced(rows: &[EasyConfigRow]) -> Vec<EasyConfigRow> {
    EASY_KEYS
        .iter()
        .filter_map(|(key, label)| {
            rows.iter()
                .find(|row| row.key.as_deref() == Some(*key))
                .map(|row| {
                    let mut row = row.clone();
                    row.label = (*label).to_string();
                    row
                })
        })
        .collect()
}

fn cycle_row_value(row: &EasyConfigRow, direction: ConfigEditDirection) -> Option<Value> {
    if row.kind == "bool" {
        return Some(Value::Bool(row.value != "true"));
    }
    if row.choices.is_empty() {
        return None;
    }
    let idx = row
        .choices
        .iter()
        .position(|choice| choice == &row.value)
        .unwrap_or(0);
    let next = match direction {
        ConfigEditDirection::Previous => (idx + row.choices.len() - 1) % row.choices.len(),
        ConfigEditDirection::Next => (idx + 1) % row.choices.len(),
    };
    Some(Value::String(row.choices[next].clone()))
}

fn read_document(path: &std::path::Path) -> (Value, Option<String>) {
    match fs::read_to_string(path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(object)) => (Value::Object(object), None),
            Ok(_) => (Value::Object(Map::new()), Some("is not an object".into())),
            Err(err) => (
                Value::Object(Map::new()),
                Some(format!("parse error: {err}")),
            ),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => (Value::Object(Map::new()), None),
        Err(err) => (
            Value::Object(Map::new()),
            Some(format!("read error: {err}")),
        ),
    }
}

fn values_from_document(document: &Value) -> BTreeMap<String, String> {
    document
        .as_object()
        .map(|object| {
            object
                .iter()
                .filter(|(key, _)| key.as_str() != "model_overrides")
                .map(|(key, value)| (key.clone(), value_to_string(value)))
                .collect()
        })
        .unwrap_or_default()
}

fn config_type_kind(ty: &hipfire_config::ConfigType) -> String {
    match ty {
        hipfire_config::ConfigType::Bool => "bool",
        hipfire_config::ConfigType::U8 => "u8",
        hipfire_config::ConfigType::U16 => "u16",
        hipfire_config::ConfigType::U32 => "u32",
        hipfire_config::ConfigType::I32 => "i32",
        hipfire_config::ConfigType::F64 => "f64",
        hipfire_config::ConfigType::String => "string",
        hipfire_config::ConfigType::Path => "path",
        hipfire_config::ConfigType::Enum { .. } => "enum",
        hipfire_config::ConfigType::Json => "json",
    }
    .to_string()
}

fn impact_label(
    mutability: hipfire_config::ConfigMutability,
    restart_impact: hipfire_config::RestartImpact,
    pending: bool,
) -> String {
    if pending {
        return "pending reload".into();
    }
    match restart_impact {
        hipfire_config::RestartImpact::ReloadModel => "reload model".into(),
        hipfire_config::RestartImpact::RestartDaemon => "restart daemon".into(),
        hipfire_config::RestartImpact::RestartService => "restart service".into(),
        hipfire_config::RestartImpact::ReconnectClients => "reconnect clients".into(),
        hipfire_config::RestartImpact::None => match mutability {
            hipfire_config::ConfigMutability::RequestOnly => "applies to new requests".into(),
            hipfire_config::ConfigMutability::RuntimeReloadable => "runtime reload".into(),
            hipfire_config::ConfigMutability::LoadTime => "reload model".into(),
            hipfire_config::ConfigMutability::Static => "restart daemon".into(),
        },
    }
}

fn impact_from_json(row: &Value) -> String {
    if row.get("pending").and_then(Value::as_bool).unwrap_or(false) {
        return "pending reload".into();
    }
    match row
        .get("restart_impact")
        .and_then(Value::as_str)
        .unwrap_or("none")
    {
        "reload_model" => "reload model".into(),
        "restart_daemon" => "restart daemon".into(),
        "restart_service" => "restart service".into(),
        "reconnect_clients" => "reconnect clients".into(),
        _ => match row
            .get("mutability")
            .and_then(Value::as_str)
            .unwrap_or("static")
        {
            "request_only" => "applies to new requests".into(),
            "runtime_reloadable" => "runtime reload".into(),
            "load_time" => "reload model".into(),
            _ => "restart daemon".into(),
        },
    }
}

fn probe_host_for(host: &str) -> String {
    match host {
        "0.0.0.0" | "" => "127.0.0.1".into(),
        "::" => "::1".into(),
        other => other.to_string(),
    }
}

fn defaults() -> BTreeMap<String, String> {
    config_schema()
        .iter()
        .filter_map(|field| {
            field
                .default
                .map(|default| (field.key.to_string(), default_to_string(default)))
        })
        .collect()
}

fn default_to_string(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value_to_string(&value))
        .unwrap_or_else(|_| raw.to_string())
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "unset".into(),
        _ => v.to_string(),
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Shared schema-driven config editor view model and local document writes.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

use crate::{
    config_path, config_schema, host_config_path, loaded_config_from_documents,
    resolve_typed_config_documents_with_layers, ConfigDiagnostic, ConfigField, ConfigLayerKind,
    ConfigMutability, ConfigScope, ConfigType, ConfigValueSource, LoadedConfig, RestartImpact,
    UnknownConfigKey,
};

#[derive(Debug, Clone, Serialize)]
pub struct ConfigEditorPaths {
    pub global: PathBuf,
    pub host: PathBuf,
}

impl Default for ConfigEditorPaths {
    fn default() -> Self {
        Self {
            global: config_path(),
            host: host_config_path(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigEditorSnapshot {
    pub source: &'static str,
    pub config_path: PathBuf,
    pub host_config_path: PathBuf,
    pub selected_model: Option<String>,
    pub diagnostics: Vec<ConfigDiagnostic>,
    pub read_error: Option<String>,
    pub host_read_error: Option<String>,
    pub unknown_keys: Vec<UnknownConfigKey>,
    pub rows: Vec<ConfigEditorRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigEditorRow {
    pub key: String,
    #[serde(rename = "type")]
    pub ty: ConfigType,
    pub enum_choices: Vec<String>,
    pub local_value: Option<Value>,
    pub active_value: Option<Value>,
    pub local_source: Option<ConfigValueSource>,
    pub active_source: Option<ConfigValueSource>,
    pub scopes: Vec<ConfigScope>,
    pub mutability: ConfigMutability,
    pub restart_impact: RestartImpact,
    pub editable_global: bool,
    pub editable_host: bool,
    pub editable_model: bool,
    pub dirty: bool,
    pub pending: bool,
    pub description: String,
    pub validation: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigEditTarget {
    Global,
    Host,
    Model { id: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ConfigEditOperation {
    Set,
    Unset,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigEditRequest {
    pub target: ConfigEditTarget,
    pub key: String,
    #[serde(default = "default_set_operation")]
    pub operation: ConfigEditOperation,
    #[serde(default)]
    pub value: Option<Value>,
    #[serde(default)]
    pub model: Option<String>,
}

fn default_set_operation() -> ConfigEditOperation {
    ConfigEditOperation::Set
}

pub fn build_config_editor_snapshot(
    active: &LoadedConfig,
    selected_model: Option<&str>,
) -> ConfigEditorSnapshot {
    let paths = ConfigEditorPaths {
        global: active.config_path.clone(),
        host: active.host_config_path.clone(),
    };
    let local = loaded_config_from_documents(
        active.config_path.clone(),
        active.raw_document.clone(),
        active.read_error.clone(),
        active.host_config_path.clone(),
        active.host_raw_document.clone(),
        active.host_read_error.clone(),
        Vec::new(),
    );
    build_snapshot_from_loaded(&paths, &local, active, selected_model)
}

pub fn build_config_editor_snapshot_from_paths(
    paths: &ConfigEditorPaths,
    active: &LoadedConfig,
    selected_model: Option<&str>,
) -> ConfigEditorSnapshot {
    let (global, global_error) = read_document(&paths.global);
    let (host, host_error) = read_document(&paths.host);
    let local = loaded_config_from_documents(
        paths.global.clone(),
        global,
        global_error,
        paths.host.clone(),
        host,
        host_error,
        Vec::new(),
    );
    build_snapshot_from_loaded(paths, &local, active, selected_model)
}

pub fn apply_config_edit(
    paths: &ConfigEditorPaths,
    request: &ConfigEditRequest,
    active: &LoadedConfig,
) -> Result<ConfigEditorSnapshot, String> {
    let field = config_schema()
        .iter()
        .find(|field| field.key == request.key)
        .ok_or_else(|| format!("unknown config key {}", request.key))?;
    if field.key == "model_overrides" {
        return Err("model_overrides is edited through model-scoped keys".to_string());
    }
    ensure_target_allowed(field, &request.target)?;

    let (path, write_target) = match &request.target {
        ConfigEditTarget::Global => (&paths.global, DocumentTarget::Global),
        ConfigEditTarget::Host => (&paths.host, DocumentTarget::Host),
        ConfigEditTarget::Model { id } => (&paths.global, DocumentTarget::Model(id.clone())),
    };

    edit_document(path, field, write_target, request)?;

    let local = load_local_for_editor(paths);
    Ok(build_snapshot_from_loaded(
        paths,
        &local,
        active,
        request.model.as_deref(),
    ))
}

pub fn encode_editor_value(field: &ConfigField, raw: &Value) -> Result<Value, String> {
    match field.ty {
        ConfigType::String | ConfigType::Path => match raw {
            Value::String(value) => Ok(Value::String(value.clone())),
            other => Ok(Value::String(value_to_editor_string(other))),
        },
        ConfigType::Enum { values } => {
            let value = raw
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value_to_editor_string(raw));
            if values.iter().any(|choice| *choice == value) {
                Ok(Value::String(value))
            } else {
                Err(format!(
                    "{} expects one of: {}",
                    field.key,
                    values.join(", ")
                ))
            }
        }
        ConfigType::Bool => match raw {
            Value::Bool(value) => Ok(Value::Bool(*value)),
            Value::String(value) => value
                .parse::<bool>()
                .map(Value::Bool)
                .map_err(|_| format!("{} expects true or false", field.key)),
            _ => Err(format!("{} expects true or false", field.key)),
        },
        ConfigType::U8 => encode_unsigned(field.key, raw, u8::MAX as u64),
        ConfigType::U16 => encode_unsigned(field.key, raw, u16::MAX as u64),
        ConfigType::U32 => encode_unsigned(field.key, raw, u32::MAX as u64),
        ConfigType::I32 => {
            let value = raw_i64(raw).ok_or_else(|| format!("{} expects an integer", field.key))?;
            if value < i32::MIN as i64 || value > i32::MAX as i64 {
                return Err(format!("{} is outside i32 range", field.key));
            }
            Ok(Value::Number(value.into()))
        }
        ConfigType::F64 => {
            let value = raw_f64(raw).ok_or_else(|| format!("{} expects a number", field.key))?;
            Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| format!("{} expects a finite number", field.key))
        }
        ConfigType::Json => match raw {
            Value::String(value) => serde_json::from_str(value).or_else(|_| Ok(raw.clone())),
            other => Ok(other.clone()),
        },
    }
}

pub fn cycle_editor_value(
    field: &ConfigField,
    current: Option<&Value>,
    forward: bool,
) -> Option<Value> {
    match field.ty {
        ConfigType::Bool => {
            let next = !current.and_then(Value::as_bool).unwrap_or(false);
            Some(Value::Bool(next))
        }
        ConfigType::Enum { values } if !values.is_empty() => {
            let current = current.and_then(Value::as_str).unwrap_or(values[0]);
            let idx = values
                .iter()
                .position(|choice| *choice == current)
                .unwrap_or(0);
            let next = if forward {
                (idx + 1) % values.len()
            } else {
                (idx + values.len() - 1) % values.len()
            };
            Some(Value::String(values[next].to_string()))
        }
        _ => None,
    }
}

fn build_snapshot_from_loaded(
    paths: &ConfigEditorPaths,
    local: &LoadedConfig,
    active: &LoadedConfig,
    selected_model: Option<&str>,
) -> ConfigEditorSnapshot {
    let local_resolved = selected_model.map(|model| {
        resolve_typed_config_documents_with_layers(
            &local.raw_document,
            &local.host_raw_document,
            Some(model),
            &[],
        )
    });
    let active_resolved = selected_model.map(|model| active.resolve_for_model(model));

    let local_values = local_resolved
        .as_ref()
        .map(|resolved| &resolved.resolution.values)
        .unwrap_or(&local.resolution.values);
    let active_values = active_resolved
        .as_ref()
        .map(|resolved| &resolved.resolution.values)
        .unwrap_or(&active.resolution.values);

    let mut diagnostics = Vec::new();
    diagnostics.extend(local.diagnostics.clone());
    if let Some(resolved) = &local_resolved {
        diagnostics.extend(resolved.diagnostics.clone());
    }
    diagnostics.extend(active.diagnostics.clone());
    if let Some(resolved) = &active_resolved {
        diagnostics.extend(resolved.diagnostics.clone());
    }

    let mut unknown_keys = local.resolution.unknown_keys.clone();
    if let Some(resolved) = &local_resolved {
        unknown_keys.extend(resolved.resolution.unknown_keys.clone());
    }

    let rows = config_schema()
        .iter()
        .filter(|field| field.key != "model_overrides")
        .map(|field| {
            let local_value = resolved_value(local_values, field.key);
            let active_value = resolved_value(active_values, field.key);
            let dirty = local_value
                .and_then(|value| value.source.as_ref())
                .is_some_and(|source| {
                    matches!(
                        source.kind,
                        ConfigLayerKind::Global
                            | ConfigLayerKind::Host
                            | ConfigLayerKind::Model
                            | ConfigLayerKind::ModelHost
                    )
                });
            ConfigEditorRow {
                key: field.key.to_string(),
                ty: field.ty,
                enum_choices: enum_choices(field),
                local_value: local_value.and_then(|value| value.value.clone()),
                active_value: active_value.and_then(|value| value.value.clone()),
                local_source: local_value.and_then(|value| value.source.clone()),
                active_source: active_value.and_then(|value| value.source.clone()),
                scopes: field.scopes.to_vec(),
                mutability: field.mutability,
                restart_impact: field.restart_impact,
                editable_global: scope_allowed(field, ConfigScope::Global),
                editable_host: scope_allowed(field, ConfigScope::Host),
                editable_model: scope_allowed(field, ConfigScope::Model),
                dirty,
                pending: local_value.and_then(|value| value.value.as_ref())
                    != active_value.and_then(|value| value.value.as_ref()),
                description: field.description.to_string(),
                validation: field.validation.map(str::to_string),
            }
        })
        .collect();

    ConfigEditorSnapshot {
        source: "config_editor",
        config_path: paths.global.clone(),
        host_config_path: paths.host.clone(),
        selected_model: selected_model.map(str::to_string),
        diagnostics,
        read_error: local.read_error.clone(),
        host_read_error: local.host_read_error.clone(),
        unknown_keys,
        rows,
    }
}

fn load_local_for_editor(paths: &ConfigEditorPaths) -> LoadedConfig {
    let (global, global_error) = read_document(&paths.global);
    let (host, host_error) = read_document(&paths.host);
    loaded_config_from_documents(
        paths.global.clone(),
        global,
        global_error,
        paths.host.clone(),
        host,
        host_error,
        Vec::new(),
    )
}

enum DocumentTarget {
    Global,
    Host,
    Model(String),
}

fn edit_document(
    path: &Path,
    field: &ConfigField,
    target: DocumentTarget,
    request: &ConfigEditRequest,
) -> Result<(), String> {
    let mut object = read_object(path)?;
    match target {
        DocumentTarget::Global | DocumentTarget::Host => edit_object(&mut object, field, request)?,
        DocumentTarget::Model(id) => {
            let overrides = object
                .entry("model_overrides".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            let overrides = overrides
                .as_object_mut()
                .ok_or_else(|| "model_overrides is not an object".to_string())?;
            let model = overrides
                .entry(id)
                .or_insert_with(|| Value::Object(Map::new()));
            let model = model
                .as_object_mut()
                .ok_or_else(|| "model override is not an object".to_string())?;
            edit_object(model, field, request)?;
        }
    }
    write_object(path, object)
}

fn edit_object(
    object: &mut Map<String, Value>,
    field: &ConfigField,
    request: &ConfigEditRequest,
) -> Result<(), String> {
    match request.operation {
        ConfigEditOperation::Unset => {
            object.remove(field.key);
            Ok(())
        }
        ConfigEditOperation::Set => {
            let raw = request
                .value
                .as_ref()
                .ok_or_else(|| "set operation requires value".to_string())?;
            object.insert(field.key.to_string(), encode_editor_value(field, raw)?);
            Ok(())
        }
    }
}

fn ensure_target_allowed(field: &ConfigField, target: &ConfigEditTarget) -> Result<(), String> {
    let allowed = match target {
        ConfigEditTarget::Global => scope_allowed(field, ConfigScope::Global),
        ConfigEditTarget::Host => {
            scope_allowed(field, ConfigScope::Host) || scope_allowed(field, ConfigScope::Global)
        }
        ConfigEditTarget::Model { .. } => scope_allowed(field, ConfigScope::Model),
    };
    if allowed {
        Ok(())
    } else {
        Err(format!(
            "{} is not valid for the selected target",
            field.key
        ))
    }
}

fn scope_allowed(field: &ConfigField, scope: ConfigScope) -> bool {
    field
        .scopes
        .iter()
        .any(|field_scope| std::mem::discriminant(field_scope) == std::mem::discriminant(&scope))
}

fn enum_choices(field: &ConfigField) -> Vec<String> {
    match field.ty {
        ConfigType::Enum { values } => values.iter().map(|value| value.to_string()).collect(),
        ConfigType::Bool => vec!["false".to_string(), "true".to_string()],
        _ => Vec::new(),
    }
}

fn resolved_value<'a>(
    values: &'a [crate::ResolvedConfigValue],
    key: &str,
) -> Option<&'a crate::ResolvedConfigValue> {
    values.iter().find(|value| value.key == key)
}

fn read_document(path: &Path) -> (Value, Option<String>) {
    match fs::read_to_string(path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(value) => (value, None),
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

fn read_object(path: &Path) -> Result<Map<String, Value>, String> {
    let (document, error) = read_document(path);
    if let Some(error) = error {
        return Err(format!("{}: {error}", path.display()));
    }
    match document {
        Value::Object(object) => Ok(object),
        _ => Err(format!("{} is not a JSON object", path.display())),
    }
}

fn write_object(path: &Path, object: Map<String, Value>) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(&Value::Object(object))
        .map_err(|err| format!("serialize {}: {err}", path.display()))?;
    fs::write(path, format!("{text}\n")).map_err(|err| format!("write {}: {err}", path.display()))
}

fn encode_unsigned(key: &str, raw: &Value, max: u64) -> Result<Value, String> {
    let value = raw_u64(raw).ok_or_else(|| format!("{key} expects an unsigned integer"))?;
    if value > max {
        return Err(format!("{key} is outside unsigned range"));
    }
    Ok(Value::Number(value.into()))
}

fn raw_u64(raw: &Value) -> Option<u64> {
    match raw {
        Value::Number(value) => value.as_u64(),
        Value::String(value) => value.parse::<u64>().ok(),
        _ => None,
    }
}

fn raw_i64(raw: &Value) -> Option<i64> {
    match raw {
        Value::Number(value) => value.as_i64(),
        Value::String(value) => value.parse::<i64>().ok(),
        _ => None,
    }
}

fn raw_f64(raw: &Value) -> Option<f64> {
    match raw {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        _ => None,
    }
}

fn value_to_editor_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;
    use crate::{loaded_config_from_document, ConfigLayer};

    fn temp_paths(name: &str) -> ConfigEditorPaths {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("hipfire-config-editor-{name}-{nonce}"));
        ConfigEditorPaths {
            global: root.join("config.json"),
            host: root.join("config.local.json"),
        }
    }

    fn row<'a>(snapshot: &'a ConfigEditorSnapshot, key: &str) -> &'a ConfigEditorRow {
        snapshot
            .rows
            .iter()
            .find(|row| row.key == key)
            .expect("row")
    }

    #[test]
    fn snapshot_rows_include_schema_metadata_and_local_active_values() {
        let active = loaded_config_from_document(
            PathBuf::from("/tmp/config.json"),
            json!({"max_tokens": 128, "kv_cache": "q8"}),
            None,
            Vec::new(),
        );

        let snapshot = build_config_editor_snapshot(&active, None);
        let max_tokens = row(&snapshot, "max_tokens");

        assert_eq!(max_tokens.local_value, Some(json!(128)));
        assert_eq!(max_tokens.active_value, Some(json!(128)));
        assert_eq!(
            max_tokens.local_source.as_ref().unwrap().kind,
            ConfigLayerKind::Global
        );
        assert_eq!(max_tokens.mutability, ConfigMutability::RequestOnly);
        assert!(max_tokens.description.contains("generated tokens"));
    }

    #[test]
    fn pending_state_detects_active_cli_override() {
        let active = loaded_config_from_document(
            PathBuf::from("/tmp/config.json"),
            json!({"max_tokens": 128}),
            None,
            vec![ConfigLayer::new(ConfigLayerKind::Cli).with_value("max_tokens", 64)],
        );

        let snapshot = build_config_editor_snapshot(&active, None);
        let max_tokens = row(&snapshot, "max_tokens");

        assert_eq!(max_tokens.local_value, Some(json!(128)));
        assert_eq!(max_tokens.active_value, Some(json!(64)));
        assert!(max_tokens.pending);
    }

    #[test]
    fn encodes_and_rejects_schema_values() {
        let kv = config_schema()
            .iter()
            .find(|field| field.key == "kv_cache")
            .unwrap();
        let port = config_schema()
            .iter()
            .find(|field| field.key == "port")
            .unwrap();
        let temp = config_schema()
            .iter()
            .find(|field| field.key == "temperature")
            .unwrap();
        let prompt_normalize = config_schema()
            .iter()
            .find(|field| field.key == "prompt_normalize")
            .unwrap();

        assert_eq!(encode_editor_value(kv, &json!("q8")).unwrap(), json!("q8"));
        assert!(encode_editor_value(kv, &json!("bad")).is_err());
        assert_eq!(
            encode_editor_value(port, &json!("12000")).unwrap(),
            json!(12000)
        );
        assert!(encode_editor_value(port, &json!("999999")).is_err());
        assert_eq!(
            encode_editor_value(temp, &json!("0.7")).unwrap(),
            json!(0.7)
        );
        assert_eq!(
            encode_editor_value(prompt_normalize, &json!("false")).unwrap(),
            json!(false)
        );
    }

    #[test]
    fn global_host_and_model_writes_land_in_expected_documents() {
        let paths = temp_paths("targets");
        let active = LoadedConfig::from_config(crate::HipfireConfig::default());

        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Global,
                key: "max_tokens".to_string(),
                operation: ConfigEditOperation::Set,
                value: Some(json!("256")),
                model: None,
            },
            &active,
        )
        .unwrap();
        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Host,
                key: "port".to_string(),
                operation: ConfigEditOperation::Set,
                value: Some(json!(12001)),
                model: None,
            },
            &active,
        )
        .unwrap();
        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Model {
                    id: "qwen".to_string(),
                },
                key: "temperature".to_string(),
                operation: ConfigEditOperation::Set,
                value: Some(json!("0.2")),
                model: Some("qwen".to_string()),
            },
            &active,
        )
        .unwrap();

        let global: Value =
            serde_json::from_str(&fs::read_to_string(&paths.global).unwrap()).unwrap();
        let host: Value = serde_json::from_str(&fs::read_to_string(&paths.host).unwrap()).unwrap();

        assert_eq!(global["max_tokens"], json!(256));
        assert_eq!(global["model_overrides"]["qwen"]["temperature"], json!(0.2));
        assert_eq!(host["port"], json!(12001));
    }

    #[test]
    fn edits_preserve_unknown_keys_and_unset_local_override() {
        let paths = temp_paths("preserve");
        fs::create_dir_all(paths.global.parent().unwrap()).unwrap();
        fs::write(
            &paths.global,
            r#"{"unknown_key": true, "max_tokens": 128, "model_overrides": {"qwen": {"unknown_model": 1, "temperature": 0.1}}}"#,
        )
        .unwrap();
        let active = LoadedConfig::from_config(crate::HipfireConfig::default());

        apply_config_edit(
            &paths,
            &ConfigEditRequest {
                target: ConfigEditTarget::Global,
                key: "max_tokens".to_string(),
                operation: ConfigEditOperation::Unset,
                value: None,
                model: None,
            },
            &active,
        )
        .unwrap();

        let global: Value =
            serde_json::from_str(&fs::read_to_string(&paths.global).unwrap()).unwrap();
        assert_eq!(global["unknown_key"], json!(true));
        assert!(global.get("max_tokens").is_none());
        assert_eq!(global["model_overrides"]["qwen"]["unknown_model"], json!(1));
    }
}

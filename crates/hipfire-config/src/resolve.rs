// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Layered config resolution with provenance.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::{ConfigField, ConfigType, Requirement};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigLayerKind {
    CompiledDefault,
    Global,
    Profile,
    Host,
    Node,
    Pool,
    Model,
    ModelHost,
    Environment,
    Cli,
    Request,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConfigLayer {
    pub kind: ConfigLayerKind,
    pub id: Option<String>,
    pub values: BTreeMap<String, Value>,
}

impl ConfigLayer {
    pub fn new(kind: ConfigLayerKind) -> Self {
        Self {
            kind,
            id: None,
            values: BTreeMap::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_value(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    pub fn from_json_object(
        kind: ConfigLayerKind,
        id: Option<impl Into<String>>,
        value: &Value,
    ) -> Option<Self> {
        let object = value.as_object()?;
        let mut layer = ConfigLayer::new(kind);
        layer.id = id.map(Into::into);
        layer.values = object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Some(layer)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct ConfigValueSource {
    pub kind: ConfigLayerKind,
    pub id: Option<String>,
}

impl ConfigValueSource {
    fn from_layer(layer: &ConfigLayer) -> Self {
        Self {
            kind: layer.kind,
            id: layer.id.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedConfigValue {
    pub key: String,
    pub value: Option<Value>,
    pub source: Option<ConfigValueSource>,
    pub overrode: Vec<ConfigValueSource>,
    pub missing_required: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnknownConfigKey {
    pub key: String,
    pub source: ConfigValueSource,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigResolution {
    pub values: Vec<ResolvedConfigValue>,
    pub unknown_keys: Vec<UnknownConfigKey>,
}

/// Environment variable name for a config key: `HIPFIRE_` + the key upper-cased.
/// One mechanical rule, no per-field table — a new schema field gets its env
/// override for free, and nobody has to remember to register it.
pub fn env_var_name_for_key(key: &str) -> String {
    format!("HIPFIRE_{}", key.to_uppercase())
}

/// Build the `Environment` layer from `HIPFIRE_*` variables.
///
/// This exists so env is a RESOLUTION SOURCE rather than a bypass. Before it,
/// subsystems read `env::var` directly and overwrote the already-resolved config
/// value in place: env silently outranked config, nothing announced it, and one
/// setting had two independent sources. Routed through here instead, an override
/// lands in the normal precedence chain (files < environment < cli/request) and
/// `hipfire config show` reports `environment` as the source.
///
/// Values are parsed AGAINST THE FIELD'S TYPE and a bad one is rejected, not
/// coerced. That matters more than it looks: every resolved value is fed to one
/// `serde_json::from_value::<HipfireConfig>`, so a single unparseable env var
/// would fail the whole struct and silently fall back to `Default` — one typo
/// would reset every other setting. Rejects are returned for the caller to
/// surface as diagnostics.
pub fn config_layer_from_env(fields: &[ConfigField]) -> (Option<ConfigLayer>, Vec<String>) {
    config_layer_from_env_with(fields, |name| std::env::var(name).ok())
}

/// Testable core of [`config_layer_from_env`].
pub fn config_layer_from_env_with(
    fields: &[ConfigField],
    lookup: impl Fn(&str) -> Option<String>,
) -> (Option<ConfigLayer>, Vec<String>) {
    let mut layer = ConfigLayer::new(ConfigLayerKind::Environment);
    let mut rejected = Vec::new();

    for field in fields {
        let name = env_var_name_for_key(field.key);
        let Some(raw) = lookup(&name) else { continue };
        match parse_env_value(&raw, &field.ty) {
            Ok(value) => {
                layer.values.insert(field.key.to_string(), value);
            }
            Err(why) => rejected.push(format!("{name}={raw} ignored: {why}")),
        }
    }

    if layer.values.is_empty() {
        (None, rejected)
    } else {
        (Some(layer), rejected)
    }
}

fn parse_env_value(raw: &str, ty: &ConfigType) -> Result<Value, String> {
    let trimmed = raw.trim();
    match ty {
        // `1`/`0` as well as `true`/`false`: the env vars this layer replaces
        // overwhelmingly used `=1`, and rejecting that would break every habit
        // and doc line that already exists.
        ConfigType::Bool => match trimmed {
            "1" | "true" | "yes" | "on" => Ok(Value::Bool(true)),
            "0" | "false" | "no" | "off" => Ok(Value::Bool(false)),
            _ => Err("want a boolean (1/0, true/false, yes/no, on/off)".to_string()),
        },
        ConfigType::U8 | ConfigType::U16 | ConfigType::U32 | ConfigType::U64 => trimmed
            .parse::<u64>()
            .map(|n| Value::from(n))
            .map_err(|_| "want a non-negative integer".to_string()),
        ConfigType::I32 => trimmed
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| "want an integer".to_string()),
        ConfigType::F64 => trimmed
            .parse::<f64>()
            .map(|n| Value::from(n))
            .map_err(|_| "want a number".to_string()),
        ConfigType::Enum { values } => {
            if values.contains(&trimmed) {
                Ok(Value::String(trimmed.to_string()))
            } else {
                Err(format!("want one of {}", values.join("|")))
            }
        }
        ConfigType::String => Ok(Value::String(raw.to_string())),
        ConfigType::Path { .. } => {
            validate_path(raw)?;
            Ok(Value::String(raw.to_string()))
        }
        ConfigType::Json => {
            serde_json::from_str::<Value>(raw).map_err(|err| format!("want valid JSON ({err})"))
        }
        // First arm that accepts wins; see `ConfigType::OneOf` on why order is
        // semantic. On total failure report every arm's expectation, not just
        // the last one's — "want a path" alone would hide the sentinels.
        ConfigType::OneOf(arms) => {
            let mut wants = Vec::new();
            for arm in *arms {
                match parse_env_value(raw, arm) {
                    Ok(value) => return Ok(value),
                    Err(want) => wants.push(want),
                }
            }
            Err(wants.join("; or "))
        }
    }
}

/// A config path must be ABSOLUTE.
///
/// Deliberately NOT an existence check: a store root is created on first use,
/// and requiring it to exist would refuse a valid config on a fresh host.
///
/// Absoluteness is what makes a sentinel typo catchable. A bare relative name
/// like `"rma"` (for `"ram"`) is a legal path, so with relatives allowed it
/// resolved silently as a directory and only the filesystem could say it was
/// not meant. It is also the right rule on its own terms: a config file is read
/// from a daemon whose working directory is not the operator's, so a relative
/// path means something different depending on how the daemon was started.
///
/// `~` is NOT accepted. Nothing in the config path expands it, so allowing it
/// would create a literal `~` directory — worse than refusing it.
pub fn validate_path(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("want an absolute path (got blank)".to_string());
    }
    if raw.contains('\0') {
        return Err("want an absolute path (contains a NUL byte)".to_string());
    }
    if trimmed.starts_with('~') {
        return Err(
            "want an absolute path (`~` is not expanded here; write the full path)".to_string(),
        );
    }
    if !trimmed.starts_with('/') {
        return Err(format!(
            "want an absolute path (`{trimmed}` is relative; it would resolve against the \
             daemon's working directory, not yours)"
        ));
    }
    Ok(())
}

/// Check an already-typed JSON value (from a config FILE, not the environment)
/// against its declared type.
///
/// `resolve_field` takes file values verbatim, so before this nothing checked
/// them: only env went through `parse_env_value`, and the later
/// `from_value::<HipfireConfig>` catches a Rust type mismatch but never a
/// domain one — `kv_cache: "kvarnn"` is a perfectly good String.
///
/// Reports, never rejects. A value that is wrong here still resolves exactly as
/// it did before; the operator gets told. Refusing would turn a warning into an
/// outage on configs that are running today.
pub fn validate_resolved_value(value: &Value, ty: &ConfigType) -> Result<(), String> {
    match (value, ty) {
        (_, ConfigType::Json) => Ok(()),
        (Value::String(text), _) => parse_env_value(text, ty).map(|_| ()),
        (Value::Bool(_), ConfigType::Bool) => Ok(()),
        (
            Value::Number(_),
            ConfigType::U8 | ConfigType::U16 | ConfigType::U32 | ConfigType::U64,
        ) if value.as_u64().is_some() => Ok(()),
        (Value::Number(_), ConfigType::I32) if value.as_i64().is_some() => Ok(()),
        (Value::Number(_), ConfigType::F64) => Ok(()),
        (_, ConfigType::OneOf(arms)) => {
            let mut wants = Vec::new();
            for arm in *arms {
                match validate_resolved_value(value, arm) {
                    Ok(()) => return Ok(()),
                    Err(want) => wants.push(want),
                }
            }
            Err(wants.join("; or "))
        }
        // Anything else is a shape mismatch the typed materialize step reports
        // with its own message; do not double-report it here.
        _ => Ok(()),
    }
}

pub fn resolve_config_layers(fields: &[ConfigField], layers: &[ConfigLayer]) -> ConfigResolution {
    let known = fields
        .iter()
        .map(|field| field.key)
        .collect::<BTreeSet<_>>();
    let mut unknown_keys = Vec::new();
    for layer in layers {
        for key in layer.values.keys() {
            if !known.contains(key.as_str()) {
                unknown_keys.push(UnknownConfigKey {
                    key: key.clone(),
                    source: ConfigValueSource::from_layer(layer),
                });
            }
        }
    }

    let mut values = fields
        .iter()
        .map(|field| resolve_field(field, layers))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.key.cmp(&b.key));
    unknown_keys.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.source.kind.cmp(&b.source.kind))
            .then_with(|| a.source.id.cmp(&b.source.id))
    });

    ConfigResolution {
        values,
        unknown_keys,
    }
}

pub fn config_layers_from_document(raw: &Value, model_tag: Option<&str>) -> Vec<ConfigLayer> {
    config_layers_from_documents(raw, None, model_tag)
}

pub fn config_layers_from_documents(
    raw: &Value,
    host_local: Option<&Value>,
    model_tag: Option<&str>,
) -> Vec<ConfigLayer> {
    let Some(object) = raw.as_object() else {
        return host_local
            .and_then(Value::as_object)
            .map(|host_object| layers_from_host_object(host_object, model_tag))
            .unwrap_or_default();
    };

    let mut global = ConfigLayer::new(ConfigLayerKind::Global);
    for (key, value) in object {
        if key != "model_overrides" {
            global.values.insert(key.clone(), value.clone());
        }
    }

    let mut layers = Vec::new();
    if !global.values.is_empty() {
        layers.push(global);
    }

    let host_object = host_local.and_then(Value::as_object);
    if let Some(host_object) = host_object {
        layers.extend(layers_from_host_object(host_object, None));
    }

    if let Some(tag) = model_tag {
        if let Some(model_values) = object
            .get("model_overrides")
            .and_then(Value::as_object)
            .and_then(|overrides| overrides.get(tag))
        {
            if let Some(layer) =
                ConfigLayer::from_json_object(ConfigLayerKind::Model, Some(tag), model_values)
            {
                if !layer.values.is_empty() {
                    layers.push(layer);
                }
            }
        }

        if let Some(model_values) = host_object
            .and_then(|object| object.get("model_overrides"))
            .and_then(Value::as_object)
            .and_then(|overrides| overrides.get(tag))
        {
            if let Some(layer) =
                ConfigLayer::from_json_object(ConfigLayerKind::ModelHost, Some(tag), model_values)
            {
                if !layer.values.is_empty() {
                    layers.push(layer);
                }
            }
        }
    }

    layers
}

fn layers_from_host_object(
    host_object: &serde_json::Map<String, Value>,
    model_tag: Option<&str>,
) -> Vec<ConfigLayer> {
    let mut layers = Vec::new();
    let mut host = ConfigLayer::new(ConfigLayerKind::Host).with_id("local");
    for (key, value) in host_object {
        if key != "model_overrides" {
            host.values.insert(key.clone(), value.clone());
        }
    }
    if !host.values.is_empty() {
        layers.push(host);
    }
    if let Some(tag) = model_tag {
        if let Some(model_values) = host_object
            .get("model_overrides")
            .and_then(Value::as_object)
            .and_then(|overrides| overrides.get(tag))
        {
            if let Some(layer) =
                ConfigLayer::from_json_object(ConfigLayerKind::ModelHost, Some(tag), model_values)
            {
                if !layer.values.is_empty() {
                    layers.push(layer);
                }
            }
        }
    }
    layers
}

fn resolve_field(field: &ConfigField, layers: &[ConfigLayer]) -> ResolvedConfigValue {
    let mut value = field.default.map(parse_default_value);
    let mut source = value.as_ref().map(|_| ConfigValueSource {
        kind: ConfigLayerKind::CompiledDefault,
        id: None,
    });
    let mut overrode = Vec::new();

    for layer in layers {
        if let Some(next) = layer.values.get(field.key) {
            if let Some(prev) = source.take() {
                overrode.push(prev);
            }
            value = Some(next.clone());
            source = Some(ConfigValueSource::from_layer(layer));
        }
    }

    let missing_required = matches!(field.requirement, Requirement::Required) && source.is_none();

    ResolvedConfigValue {
        key: field.key.to_string(),
        value,
        source,
        overrode,
        missing_required,
    }
}

fn parse_default_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{resolve_config_layers, ConfigLayer, ConfigLayerKind};
    use crate::schema::config_schema;

    fn value<'a>(
        resolution: &'a super::ConfigResolution,
        key: &str,
    ) -> &'a super::ResolvedConfigValue {
        resolution
            .values
            .iter()
            .find(|value| value.key == key)
            .expect("resolved key")
    }

    #[test]
    fn resolves_defaults_without_layers() {
        let resolution = resolve_config_layers(config_schema(), &[]);
        let max_tokens = value(&resolution, "max_tokens");

        assert_eq!(max_tokens.value, Some(json!(512)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::CompiledDefault)
        );
        assert!(max_tokens.overrode.is_empty());
        assert!(resolution.unknown_keys.is_empty());
    }

    #[test]
    fn higher_layers_override_lower_layers_with_provenance() {
        let global = ConfigLayer::new(ConfigLayerKind::Global).with_value("max_tokens", 256);
        let model = ConfigLayer::new(ConfigLayerKind::Model)
            .with_id("qwen3.5:9b")
            .with_value("max_tokens", 1024);
        let request = ConfigLayer::new(ConfigLayerKind::Request).with_value("max_tokens", 64);

        let resolution = resolve_config_layers(config_schema(), &[global, model, request]);
        let max_tokens = value(&resolution, "max_tokens");

        assert_eq!(max_tokens.value, Some(json!(64)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Request)
        );
        assert_eq!(
            max_tokens
                .overrode
                .iter()
                .map(|source| source.kind)
                .collect::<Vec<_>>(),
            vec![
                ConfigLayerKind::CompiledDefault,
                ConfigLayerKind::Global,
                ConfigLayerKind::Model,
            ]
        );
    }

    #[test]
    fn reports_unknown_layer_keys() {
        let layer = ConfigLayer::new(ConfigLayerKind::Host)
            .with_id("strix-halo-01")
            .with_value("vision.max_cores", 6);

        let resolution = resolve_config_layers(config_schema(), &[layer]);

        assert_eq!(resolution.unknown_keys.len(), 1);
        assert_eq!(resolution.unknown_keys[0].key, "vision.max_cores");
        assert_eq!(
            resolution.unknown_keys[0].source.kind,
            ConfigLayerKind::Host
        );
        assert_eq!(
            resolution.unknown_keys[0].source.id.as_deref(),
            Some("strix-halo-01")
        );
    }

    #[test]
    fn builds_global_and_model_layers_from_config_document() {
        let raw = json!({
            "max_tokens": 256,
            "temperature": 0.4,
            "model_overrides": {
                "qwen3.5:9b": {
                    "temperature": 0.1,
                    "kv_cache": "q8"
                },
                "other": {
                    "temperature": 0.8
                }
            }
        });

        let layers = super::config_layers_from_document(&raw, Some("qwen3.5:9b"));
        let resolution = resolve_config_layers(config_schema(), &layers);

        let temperature = value(&resolution, "temperature");
        assert_eq!(temperature.value, Some(json!(0.1)));
        assert_eq!(
            temperature.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Model)
        );
        assert_eq!(
            temperature
                .source
                .as_ref()
                .and_then(|source| source.id.as_deref()),
            Some("qwen3.5:9b")
        );

        let max_tokens = value(&resolution, "max_tokens");
        assert_eq!(max_tokens.value, Some(json!(256)));
        assert_eq!(
            max_tokens.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Global)
        );

        let model_overrides = value(&resolution, "model_overrides");
        assert_eq!(model_overrides.value, Some(json!({})));
        assert_eq!(
            model_overrides.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::CompiledDefault)
        );
    }
}

#[cfg(test)]
mod env_layer_tests {
    use super::{config_layer_from_env_with, env_var_name_for_key, ConfigLayerKind};
    use crate::schema::config_schema;
    use std::collections::HashMap;

    fn layer_from(pairs: &[(&str, &str)]) -> (Option<super::ConfigLayer>, Vec<String>) {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        config_layer_from_env_with(config_schema(), |name| map.get(name).cloned())
    }

    #[test]
    fn name_rule_is_mechanical() {
        assert_eq!(
            env_var_name_for_key("prefill_block"),
            "HIPFIRE_PREFILL_BLOCK"
        );
    }

    #[test]
    fn typed_values_land_in_an_environment_layer() {
        let (layer, rejected) = layer_from(&[
            ("HIPFIRE_PORT", "18080"),
            ("HIPFIRE_KV_CACHE", "kvarn8"),
            ("HIPFIRE_PREFILL_PROFILE", "1"),
        ]);
        let layer = layer.expect("env layer");
        assert_eq!(layer.kind, ConfigLayerKind::Environment);
        assert_eq!(layer.values["port"], 18080);
        assert_eq!(layer.values["kv_cache"], "kvarn8");
        // `=1` for a bool: the env vars this layer replaces used that spelling.
        assert_eq!(layer.values["prefill_profile"], true);
        assert!(rejected.is_empty(), "{rejected:?}");
    }

    #[test]
    fn a_bad_value_is_rejected_alone_and_never_poisons_the_rest() {
        // The whole resolved map goes through ONE from_value::<HipfireConfig>, so
        // an unparseable value that got through would fail the entire struct and
        // fall back to Default -- one typo silently resetting every setting.
        let (layer, rejected) = layer_from(&[
            ("HIPFIRE_PORT", "not-a-port"),
            ("HIPFIRE_KV_CACHE", "kvarn9"),
            ("HIPFIRE_HOST", "0.0.0.0"),
        ]);
        let layer = layer.expect("env layer");
        assert!(!layer.values.contains_key("port"));
        assert!(!layer.values.contains_key("kv_cache"));
        assert_eq!(layer.values["host"], "0.0.0.0");
        assert_eq!(rejected.len(), 2, "{rejected:?}");
        // The message names the legal values, so a typo tells you the fix.
        assert!(rejected
            .iter()
            .any(|m| m.contains("HIPFIRE_KV_CACHE") && m.contains("kvarn8")));
    }

    #[test]
    fn no_env_means_no_layer() {
        let (layer, rejected) = layer_from(&[]);
        assert!(layer.is_none());
        assert!(rejected.is_empty());
    }
}

use clap::Args;
use hipfire_config::{ConfigLayer, ConfigLayerKind, LoadedConfig};
use serde_json::json;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire serve\n  hipfire serve --host 0.0.0.0 --port 11435\n  hipfire serve --model Qwen3.5-30B-A3B\n"
)]
pub struct ServeArgs {
    /// Override bind host
    #[arg(long)]
    pub host: Option<String>,
    /// Override bind port
    #[arg(long, short)]
    pub port: Option<u16>,
    /// Default model name, shorthand, alias, or path for requests that omit model
    #[arg(long, short)]
    pub model: Option<String>,
    /// Log full raw chat requests and raw model replies.
    #[arg(long)]
    pub debug_chat: bool,
}

pub async fn run(args: ServeArgs, config: LoadedConfig) -> anyhow::Result<()> {
    let mut cli_layer = ConfigLayer::new(ConfigLayerKind::Cli);
    if let Some(h) = args.host {
        cli_layer.values.insert("host".to_string(), json!(h));
    }
    if let Some(p) = args.port {
        cli_layer.values.insert("port".to_string(), json!(p));
    }
    if let Some(m) = args.model {
        cli_layer
            .values
            .insert("default_model".to_string(), json!(m));
    }
    if args.debug_chat {
        std::env::set_var("HIPFIRE_DEBUG_CHAT", "1");
    }
    hipfire_server::serve_loaded(config.with_additional_layer(cli_layer)).await
}

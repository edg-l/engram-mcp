pub mod dispatch;
pub mod filter;
pub mod install;
pub mod payload;
pub mod redact;
pub mod transcript;

use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    UserPromptSubmit,
    PostToolUse,
    Stop,
    PreCompact,
    SessionEnd,
    SubagentStop,
}

impl FromStr for HookEvent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SessionStart" => Ok(HookEvent::SessionStart),
            "UserPromptSubmit" => Ok(HookEvent::UserPromptSubmit),
            "PostToolUse" => Ok(HookEvent::PostToolUse),
            "Stop" => Ok(HookEvent::Stop),
            "PreCompact" => Ok(HookEvent::PreCompact),
            "SessionEnd" => Ok(HookEvent::SessionEnd),
            "SubagentStop" => Ok(HookEvent::SubagentStop),
            other => Err(format!("unknown hook event: {}", other)),
        }
    }
}

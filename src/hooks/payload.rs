use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct SessionStartPayload {}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct UserPromptSubmitPayload {
    pub prompt: Option<String>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct PostToolUsePayload {
    pub tool_name: Option<String>,
    pub tool_input: Option<serde_json::Value>,
    pub tool_response: Option<serde_json::Value>,
    // UNVERIFIED: likely not a top-level field; failure data lives inside tool_response
    pub exit_code: Option<i32>,
    pub session_id: Option<String>,
    pub transcript_path: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct StopPayload {
    pub stop_hook_active: Option<bool>,
    pub response: Option<String>,
    pub session_id: Option<String>,
    pub transcript_path: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct PreCompactPayload {
    pub compaction_reason: Option<String>,
    pub transcript_path: Option<String>,
    pub current_context_usage: Option<i64>,
    pub context_limit: Option<i64>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct SessionEndPayload {
    pub reason: Option<String>,
    pub transcript_path: Option<String>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct SubagentStopPayload {
    pub agent_type: Option<String>,
    // UNVERIFIED: may use `response` like StopPayload
    pub last_assistant_message: Option<String>,
    pub session_id: Option<String>,
    pub transcript_path: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_session_start_payload() {
        let json = r#"{}"#;
        let p: SessionStartPayload = serde_json::from_str(json).unwrap();
        let _ = p;
    }

    #[test]
    fn deserialize_user_prompt_submit_payload() {
        let json = r#"{"prompt":"hello","session_id":"abc","cwd":"/home/user"}"#;
        let p: UserPromptSubmitPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.prompt.as_deref(), Some("hello"));
        assert_eq!(p.session_id.as_deref(), Some("abc"));
        assert_eq!(p.cwd.as_deref(), Some("/home/user"));
    }

    #[test]
    fn deserialize_post_tool_use_payload() {
        let json = r#"{"tool_name":"Bash","exit_code":0,"session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/home/user"}"#;
        let p: PostToolUsePayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.tool_name.as_deref(), Some("Bash"));
        assert_eq!(p.exit_code, Some(0));
        assert!(p.tool_input.is_none());
        assert_eq!(p.session_id.as_deref(), Some("s1"));
        assert_eq!(p.transcript_path.as_deref(), Some("/tmp/t.jsonl"));
        assert_eq!(p.cwd.as_deref(), Some("/home/user"));
    }

    #[test]
    fn deserialize_stop_payload() {
        let json = r#"{"stop_hook_active":true,"response":"done","session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/home/user"}"#;
        let p: StopPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.stop_hook_active, Some(true));
        assert_eq!(p.response.as_deref(), Some("done"));
        assert_eq!(p.session_id.as_deref(), Some("s1"));
        assert_eq!(p.transcript_path.as_deref(), Some("/tmp/t.jsonl"));
        assert_eq!(p.cwd.as_deref(), Some("/home/user"));
    }

    #[test]
    fn deserialize_pre_compact_payload() {
        let json = r#"{"compaction_reason":"context_full","transcript_path":"/tmp/t.jsonl","current_context_usage":90000,"context_limit":100000,"session_id":"s1","cwd":"/home/user"}"#;
        let p: PreCompactPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.compaction_reason.as_deref(), Some("context_full"));
        assert_eq!(p.transcript_path.as_deref(), Some("/tmp/t.jsonl"));
        assert_eq!(p.current_context_usage, Some(90000));
        assert_eq!(p.context_limit, Some(100000));
        assert_eq!(p.session_id.as_deref(), Some("s1"));
        assert_eq!(p.cwd.as_deref(), Some("/home/user"));
    }

    #[test]
    fn deserialize_session_end_payload() {
        let json = r#"{"reason":"user_exit","transcript_path":"/tmp/t.jsonl","session_id":"s1","cwd":"/home/user"}"#;
        let p: SessionEndPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.reason.as_deref(), Some("user_exit"));
        assert_eq!(p.transcript_path.as_deref(), Some("/tmp/t.jsonl"));
        assert_eq!(p.session_id.as_deref(), Some("s1"));
        assert_eq!(p.cwd.as_deref(), Some("/home/user"));
    }

    #[test]
    fn deserialize_subagent_stop_payload() {
        let json = r#"{"agent_type":"plan-implementer","last_assistant_message":"Phase complete","session_id":"s1","transcript_path":"/tmp/t.jsonl"}"#;
        let p: SubagentStopPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.agent_type.as_deref(), Some("plan-implementer"));
        assert_eq!(p.last_assistant_message.as_deref(), Some("Phase complete"));
        assert_eq!(p.session_id.as_deref(), Some("s1"));
        assert_eq!(p.transcript_path.as_deref(), Some("/tmp/t.jsonl"));
    }
}

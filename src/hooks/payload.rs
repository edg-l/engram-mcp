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
    pub exit_code: Option<i32>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct StopPayload {
    pub stop_hook_active: Option<bool>,
    pub last_assistant_message: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct PreCompactPayload {
    pub trigger: Option<String>,
    pub custom_instructions: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct SessionEndPayload {
    pub reason: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(default)]
pub struct SubagentStopPayload {
    pub agent_type: Option<String>,
    pub last_assistant_message: Option<String>,
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
        let json = r#"{"tool_name":"Bash","exit_code":0}"#;
        let p: PostToolUsePayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.tool_name.as_deref(), Some("Bash"));
        assert_eq!(p.exit_code, Some(0));
        assert!(p.tool_input.is_none());
    }

    #[test]
    fn deserialize_stop_payload() {
        let json = r#"{"stop_hook_active":true,"last_assistant_message":"done"}"#;
        let p: StopPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.stop_hook_active, Some(true));
        assert_eq!(p.last_assistant_message.as_deref(), Some("done"));
    }

    #[test]
    fn deserialize_pre_compact_payload() {
        let json = r#"{"trigger":"manual","custom_instructions":"focus on bugs"}"#;
        let p: PreCompactPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.trigger.as_deref(), Some("manual"));
        assert_eq!(p.custom_instructions.as_deref(), Some("focus on bugs"));
    }

    #[test]
    fn deserialize_session_end_payload() {
        let json = r#"{"reason":"user_exit"}"#;
        let p: SessionEndPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.reason.as_deref(), Some("user_exit"));
    }

    #[test]
    fn deserialize_subagent_stop_payload() {
        let json = r#"{"agent_type":"plan-implementer","last_assistant_message":"Phase complete"}"#;
        let p: SubagentStopPayload = serde_json::from_str(json).unwrap();
        assert_eq!(p.agent_type.as_deref(), Some("plan-implementer"));
        assert_eq!(p.last_assistant_message.as_deref(), Some("Phase complete"));
    }
}

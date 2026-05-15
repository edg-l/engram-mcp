use engram_mcp::tools::schemas::{ToolProfile, get_tool_definitions, get_tool_definitions_for};

#[test]
fn minimal_profile_has_three_tools() {
    let tools = get_tool_definitions_for(ToolProfile::Minimal);
    assert_eq!(tools.len(), 3);
    let names: std::collections::HashSet<String> =
        tools.iter().map(|t| t.name.to_string()).collect();
    let expected: std::collections::HashSet<String> =
        ["memory_context", "memory_store", "handoff_resume"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    assert_eq!(names, expected);
}

#[test]
fn core_profile_has_eleven_tools() {
    let tools = get_tool_definitions_for(ToolProfile::Core);
    assert_eq!(tools.len(), 11);
    let names: std::collections::HashSet<String> =
        tools.iter().map(|t| t.name.to_string()).collect();
    let expected: std::collections::HashSet<String> = [
        "memory_context",
        "memory_store",
        "handoff_resume",
        "memory_query",
        "memory_update",
        "memory_delete",
        "memory_link",
        "memory_graph",
        "handoff_create",
        "memory_store_batch",
        "memory_delete_batch",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(names, expected);
}

#[test]
fn full_profile_matches_default() {
    let full = get_tool_definitions_for(ToolProfile::Full);
    assert_eq!(full.len(), 18);
    let full_names: std::collections::HashSet<String> =
        full.iter().map(|t| t.name.to_string()).collect();
    let all = get_tool_definitions();
    let all_names: std::collections::HashSet<String> =
        all.iter().map(|t| t.name.to_string()).collect();
    assert_eq!(full_names, all_names);
}

#[test]
fn profile_parses_case_insensitive() {
    assert_eq!("FULL".parse::<ToolProfile>().unwrap(), ToolProfile::Full);
    assert_eq!("Full".parse::<ToolProfile>().unwrap(), ToolProfile::Full);
    assert_eq!("full".parse::<ToolProfile>().unwrap(), ToolProfile::Full);

    assert_eq!("CORE".parse::<ToolProfile>().unwrap(), ToolProfile::Core);
    assert_eq!("Core".parse::<ToolProfile>().unwrap(), ToolProfile::Core);
    assert_eq!("core".parse::<ToolProfile>().unwrap(), ToolProfile::Core);

    assert_eq!(
        "MINIMAL".parse::<ToolProfile>().unwrap(),
        ToolProfile::Minimal
    );
    assert_eq!(
        "Minimal".parse::<ToolProfile>().unwrap(),
        ToolProfile::Minimal
    );
    assert_eq!(
        "minimal".parse::<ToolProfile>().unwrap(),
        ToolProfile::Minimal
    );

    assert!("bogus".parse::<ToolProfile>().is_err());
    assert!("".parse::<ToolProfile>().is_err());
}

//! Claude Code passes on only the first 4096 characters of an MCP server's instructions
//! and drops the rest without telling the agent. Keep them under that, with headroom.

#[test]
fn instructions_fit_what_clients_pass_on() {
    let instructions = include_str!("../src/instructions.md");
    assert!(
        instructions.chars().count() < 4000,
        "src/instructions.md is {} chars; move detail into tool descriptions or src/guide.md",
        instructions.chars().count()
    );
    assert!(instructions.contains("engram://guide"));
}

"""Terminal guidance stays separate from channel and worker prompts."""

from types import SimpleNamespace

from agent import system_prompt
from agent.prompt_builder import build_skills_system_prompt


def test_terminal_review_does_not_depend_on_skills(monkeypatch):
    monkeypatch.setattr(system_prompt, "owned_kanban_task", lambda: False)
    agent = SimpleNamespace(platform="cli", valid_tool_names={"terminal"}, model="test",
                            _tool_use_enforcement=False, _execution_guidance=False)
    guidance = " ".join(filter(None, system_prompt._guidance_parts(agent)))
    assert "inspect the actual change" in guidance
    assert "`Review:`" in guidance and "`Verification:`" in guidance
    agent.platform = "telegram"
    assert "inspect the actual change" not in " ".join(filter(None, system_prompt._guidance_parts(agent)))
    agent.platform = "cli"
    monkeypatch.setattr(system_prompt, "owned_kanban_task", lambda: True)
    assert "inspect the actual change" not in " ".join(filter(None, system_prompt._guidance_parts(agent)))


def test_skill_routing_cache_separates_terminal_from_other_surfaces(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    skill = tmp_path / "skills" / "misc" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill.\n---\n# Demo\n")
    args = {"available_tools": {"skill_view"}, "skills_dir_override": tmp_path / "skills"}
    terminal = build_skills_system_prompt(**args, terminal_guidance=True)
    channel = build_skills_system_prompt(**args, terminal_guidance=False)
    assert "Use a named skill when the user requests it" in terminal
    assert "even partially relevant" in channel
    assert build_skills_system_prompt(**args, terminal_guidance=True) == terminal

    monkeypatch.setattr(system_prompt, "_agent_skills_dir", lambda _agent: tmp_path / "skills")
    monkeypatch.setattr(system_prompt, "owned_kanban_task", lambda: False)
    agent = SimpleNamespace(platform="cli", valid_tool_names={"skill_view"})
    assert "Use a named skill when the user requests it" in system_prompt._skills_prompt(agent)
    agent.platform = "telegram"
    assert "even partially relevant" in system_prompt._skills_prompt(agent)
    agent.platform = "cli"
    monkeypatch.setattr(system_prompt, "owned_kanban_task", lambda: True)
    assert "even partially relevant" in system_prompt._skills_prompt(agent)

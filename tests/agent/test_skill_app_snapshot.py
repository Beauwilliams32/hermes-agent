"""Application requirements survive disk and in-process skill-index reuse."""

import json
import sys

from hermes_platform import declaration


def test_application_gate_rechecks_snapshot_without_losing_description(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    monkeypatch.setattr(declaration, "_REGISTRY", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    skills = tmp_path / "skills"
    skill = skills / "app-guide" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: app-guide\ndescription: Application instructions.\n"
        "requires_apps: [thing]\n---\nHelp with the application.\n", encoding="utf-8",
    )
    app = tmp_path / "application.exe"
    declaration.register("thing", declaration.parse_declaration(
        "thing", {sys.platform: {"presence": "executable", "location": str(app)}},
        {"app": True}, where="test-plugin/plugin.yaml",
    ))
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        def build():
            return pb._build_skills_system_prompt_inner(skills, [], None, None, None)

        assert "app-guide" not in build()
        snapshot = pb._load_skills_snapshot(skills)
        assert snapshot is not None
        assert snapshot["skills"][0]["requires_apps"] == ["thing"]
        app.write_text("presence fixture", encoding="utf-8")
        assert "app-guide: Application instructions." in build()
        app.unlink()
        assert "app-guide" not in build()
        pb.clear_skills_system_prompt_cache()
        assert "app-guide" not in build()
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_old_skill_snapshot_rebuilds_with_category_descriptions(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    skills = tmp_path / "skills"
    category = skills / "photography"
    skill = category / "composition" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: composition\ndescription: Compose a strong frame.\n---\nSkill body.\n",
        encoding="utf-8",
    )
    (category / "DESCRIPTION.md").write_text(
        "---\ndescription: Photography workflows.\n---\n",
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "snapshot.json"
    monkeypatch.setattr(pb, "_skills_prompt_snapshot_path", lambda: snapshot_path)
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    snapshot_path.write_text(json.dumps({
        "version": 4,
        "manifest": pb._build_skills_manifest(skills),
        "skills": [],
    }), encoding="utf-8")

    prompt = pb._build_skills_system_prompt_inner(skills, [], None, None, None, terminal_guidance=True)

    assert "photography: Photography workflows." in prompt
    refreshed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert refreshed["category_descriptions"]["photography"] == "Photography workflows."

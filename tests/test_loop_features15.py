import asyncio

from wallbreaker.cli import main
from wallbreaker.config import Config, Endpoint, doctor_report


def test_doctor_all_ok():
    ep = Endpoint("p", "openai", "http://x", "m", api_key="k")
    tgt = Endpoint("target", "openai", "http://y", "mt", api_key="k2")
    jd = Endpoint("judge", "openai", "http://z", "mj", api_key="k3")
    cfg = Config(default_profile="p", profiles={"p": ep}, target=tgt, judge=jd)
    report, ok = doctor_report(cfg)
    assert ok
    assert "READY" in report


def test_doctor_flags_missing_default():
    ep = Endpoint("p", "openai", "http://x", "m", api_key="k")
    cfg = Config(default_profile="ghost", profiles={"p": ep})
    report, ok = doctor_report(cfg)
    assert not ok
    assert "NOT READY" in report
    assert "default_profile 'ghost'" in report


def test_doctor_flags_missing_key(monkeypatch):
    monkeypatch.delenv("NEEDED_KEY", raising=False)
    ep = Endpoint("p", "openai", "http://x", "m", api_key_env="NEEDED_KEY")
    cfg = Config(default_profile="p", profiles={"p": ep})
    report, ok = doctor_report(cfg)
    assert not ok
    assert "no key" in report


def test_doctor_accepts_keyless_codex_roles(monkeypatch):
    import wallbreaker.providers.codex as codex

    monkeypatch.setattr(codex, "codex_login_status", lambda: (True, "Logged in for test"))
    brain = Endpoint("brain", "codex", "", "gpt-5.6-sol", reasoning_effort="max")
    target = Endpoint("target", "codex", "", "gpt-5.6-luna", reasoning_effort="low")
    judge = Endpoint("judge", "codex", "", "gpt-5.6-luna", reasoning_effort="high")
    cfg = Config(
        default_profile="brain",
        profiles={"brain": brain},
        target=target,
        judge=judge,
    )
    report, ok = doctor_report(cfg)
    assert ok
    assert "Logged in for test" in report


def test_doctor_rejects_unavailable_codex_login(monkeypatch):
    import wallbreaker.providers.codex as codex

    monkeypatch.setattr(codex, "codex_login_status", lambda: (False, "codex CLI not found"))
    brain = Endpoint("brain", "codex", "", "gpt-5.6-sol")
    report, ok = doctor_report(Config(default_profile="brain", profiles={"brain": brain}))
    assert not ok
    assert "codex CLI not found" in report


def test_doctor_notes_missing_target_and_judge():
    ep = Endpoint("p", "openai", "http://x", "m", api_key="k")
    cfg = Config(default_profile="p", profiles={"p": ep})
    report, ok = doctor_report(cfg)
    assert ok  # notes, not failures
    assert "no [target]" in report
    assert "no [judge]" in report


def test_cli_check_returns_code(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'default_profile = "p"\n[profiles.p]\nprotocol="openai"\n'
        'base_url="http://x"\nmodel="m"\napi_key="k"\n',
        encoding="utf-8",
    )
    rc = main(["check", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Wallbreaker config check" in out


def _build_app():
    from wallbreaker.prompts import DEFAULT_SYSTEM
    from wallbreaker.tui.app import RthApp

    ep = Endpoint("t", "openai", "http://x", "m")
    cfg = Config(default_profile="t", profiles={"t": ep}, target=ep)
    return RthApp(cfg, ep, DEFAULT_SYSTEM, prefs={"log": False})


def test_help_filter_mounts():
    async def run():
        app = _build_app()
        async with app.run_test() as pilot:
            before = len(app.query_one("#log").children)
            app._handle_command("/help report")
            app._handle_command("/help")
            await pilot.pause()
            assert len(app.query_one("#log").children) >= before + 2

    asyncio.run(run())

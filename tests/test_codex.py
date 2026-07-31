import asyncio
import json
import os
from types import SimpleNamespace

import pytest

import wallbreaker.providers.codex as codex
from wallbreaker.agent.messages import (
    ReasoningDelta,
    StopEvent,
    TextDelta,
    ToolUseEvent,
    UsageEvent,
    user,
)
from wallbreaker.config import Config, ConfigError, _endpoint_from_table, doctor_report
from wallbreaker.providers.base import ProviderError
from wallbreaker.providers.factory import build_provider


def _jsonl(*events: dict) -> bytes:
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


def _success(text: str = "ok", *, usage: dict | None = None) -> bytes:
    return _jsonl(
        {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "agent_message", "text": text},
        },
        {"type": "turn.completed", "usage": usage or {}},
    )


class _FakeProc:
    def __init__(self, out: bytes = b"", rc: int | None = 0, err: bytes = b""):
        self._out = out
        self._err = err
        self.returncode = rc
        self.pid = None
        self.stdin_data: bytes | None = None
        self.killed = False

    async def communicate(self, data=None):
        self.stdin_data = data
        return self._out, self._err

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def _patch_cli(
    monkeypatch,
    out: bytes = b"",
    *,
    rc: int | None = 0,
    err: bytes = b"",
    missing: bool = False,
    proc: _FakeProc | None = None,
):
    captured: dict = {}
    fake = proc or _FakeProc(out=out, rc=rc, err=err)

    async def _exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["proc"] = fake
        if missing:
            raise FileNotFoundError
        return fake

    monkeypatch.setattr(codex.asyncio, "create_subprocess_exec", _exec)
    return captured


def _ep(**overrides):
    table = {
        "protocol": "codex",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
    }
    table.update(overrides)
    return _endpoint_from_table("codex-brain", table, require_model=True)


def _collect(provider, messages=None, **kwargs):
    async def go():
        return [
            event
            async for event in provider.stream(messages or [user("hello")], **kwargs)
        ]

    return asyncio.run(go())


# ---- config / factory --------------------------------------------------------


def test_config_and_factory_support_keyless_codex_with_reasoning_effort(monkeypatch):
    monkeypatch.setattr(codex, "codex_login_status", lambda: (True, "Logged in for test"))
    endpoint = _ep(reasoning_effort="MAX")
    assert endpoint.base_url == ""
    assert endpoint.resolved_key() == ""
    assert endpoint.reasoning_effort == "max"
    assert isinstance(build_provider(endpoint), codex.CodexProvider)

    report, ready = doctor_report(
        Config(default_profile="codex-brain", profiles={"codex-brain": endpoint})
    )
    assert ready is True
    assert "Logged in for test" in report


def test_config_rejects_missing_model_and_invalid_reasoning_effort():
    with pytest.raises(ConfigError, match="missing keys: model"):
        _endpoint_from_table(
            "target", {"protocol": "codex"}, require_model=True
        )
    with pytest.raises(ConfigError, match="invalid reasoning_effort"):
        _ep(reasoning_effort="turbo")


# ---- process boundary --------------------------------------------------------


def test_cli_argv_env_system_model_effort_and_stdin(monkeypatch):
    monkeypatch.setenv("WALLBREAKER_CODEX_BIN", "/opt/test/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak-either")
    monkeypatch.setenv("HOME", "/tmp/test-home")
    captured = _patch_cli(monkeypatch, _success("answer"))

    provider = build_provider(_ep())
    events = _collect(provider, [user("first"), user("final prompt")], system="OPERATOR")
    assert any(isinstance(event, TextDelta) and event.text == "answer" for event in events)

    args = list(captured["args"])
    kwargs = captured["kwargs"]
    assert args[0] == "/opt/test/codex"
    assert ["-m", "gpt-5.6-sol"] == args[args.index("-m") : args.index("-m") + 2]
    assert "model_reasoning_effort=\"max\"" in args
    assert "--ephemeral" in args
    assert "--ignore-user-config" in args
    assert "--ignore-rules" in args
    assert ["--sandbox", "read-only"] == args[
        args.index("--sandbox") : args.index("--sandbox") + 2
    ]
    assert args[-1] == "-"

    developer_arg = next(
        arg for arg in args if arg.startswith("developer_instructions=")
    )
    developer_instructions = json.loads(developer_arg.split("=", 1)[1])
    assert developer_instructions.startswith("OPERATOR")
    assert "CODEX CLI ADAPTER BOUNDARY" in developer_instructions
    assert "# HOW YOU ACT" not in developer_instructions

    workdir = args[args.index("-C") + 1]
    assert kwargs["cwd"] == workdir
    assert not os.path.exists(workdir)
    assert kwargs["env"]["HOME"] == "/tmp/test-home"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert "WALLBREAKER_CODEX_BIN" not in kwargs["env"]
    assert captured["proc"].stdin_data.decode() == (
        "USER: first\n\nUSER: final prompt"
    )
    if os.name == "posix":
        assert kwargs["start_new_session"] is True


# ---- JSONL normalization -----------------------------------------------------


def test_last_message_reasoning_usage_and_cache_are_normalized(monkeypatch):
    captured = _patch_cli(
        monkeypatch,
        _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            {
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "consider "},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "draft"},
            },
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "ignored"},
            },
            {
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "done"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "final"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 17,
                    "cached_input_tokens": 5,
                    "output_tokens": 3,
                    "cache_write_input_tokens": 2,
                },
            },
        ),
    )
    events = _collect(build_provider(_ep()))

    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == [
        "consider ",
        "done",
    ]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["final"]
    usage = next(event for event in events if isinstance(event, UsageEvent))
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    ) == (17, 3, 5, 2)
    assert events[-1] == StopEvent("end_turn")
    assert captured["proc"].returncode == 0


def test_tool_tags_become_tool_events_and_protocol_is_in_system(monkeypatch):
    reply = (
        "Checking.\n"
        '<tool_call>{"name":"query_target","input":{"prompt":"hi"}}</tool_call>\n'
        '<tool_call>{"name":"finish","input":{"summary":"done"}}</tool_call>'
    )
    captured = _patch_cli(monkeypatch, _success(reply))
    tools = [
        {
            "name": "query_target",
            "description": "ask target",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "finish",
            "description": "finish",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    events = _collect(build_provider(_ep()), tools=tools, system="OP")

    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "Checking."
    ]
    calls = [event for event in events if isinstance(event, ToolUseEvent)]
    assert [(call.id, call.name) for call in calls] == [
        ("codex_0", "query_target"),
        ("codex_1", "finish"),
    ]
    assert calls[0].input == {"prompt": "hi"}
    assert events[-1] == StopEvent("tool_use")

    developer_arg = next(
        arg
        for arg in captured["args"]
        if arg.startswith("developer_instructions=")
    )
    system = json.loads(developer_arg.split("=", 1)[1])
    assert "HARNESS TOOLS" in system
    assert "query_target" in system


def test_retry_warning_and_item_error_are_nonfatal_after_success(monkeypatch):
    _patch_cli(
        monkeypatch,
        _jsonl(
            {"type": "error", "message": "transient retry"},
            {
                "type": "item.completed",
                "item": {"type": "error", "message": "warning only"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "recovered"},
            },
            {"type": "turn.completed", "usage": {}},
        ),
    )
    assert asyncio.run(build_provider(_ep()).complete([user("retry")])) == "recovered"


# ---- fatal paths -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "rc", "stderr", "match"),
    [
        (
            _jsonl({"type": "turn.failed", "error": {"message": "quota exhausted"}}),
            0,
            b"",
            "turn failed: quota exhausted",
        ),
        (
            _jsonl(
                {"type": "error", "message": "retries exhausted"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "stale"},
                },
                {"type": "turn.completed", "usage": {}},
            ),
            7,
            b"stderr fallback",
            "exited 7: retries exhausted",
        ),
        (
            _jsonl(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "unterminated"},
                }
            ),
            0,
            b"connection closed",
            "without turn.completed: connection closed",
        ),
        (b"not-json\n", 0, b"", "malformed JSONL"),
        (
            _jsonl({"type": "turn.completed", "usage": {}}),
            0,
            b"",
            "without an agent message",
        ),
    ],
)
def test_fatal_jsonl_and_process_outcomes_raise(
    monkeypatch, raw, rc, stderr, match
):
    _patch_cli(monkeypatch, raw, rc=rc, err=stderr)
    with pytest.raises(ProviderError, match=match):
        asyncio.run(build_provider(_ep()).complete([user("x")]))


def test_missing_binary_raises_clear_provider_error(monkeypatch):
    _patch_cli(monkeypatch, missing=True)
    with pytest.raises(ProviderError, match="codex CLI not found"):
        asyncio.run(build_provider(_ep()).complete([user("x")]))


def test_timeout_kills_and_reaps_process(monkeypatch):
    class _HungProc(_FakeProc):
        def __init__(self):
            super().__init__(rc=None)
            self.waited = False

        async def communicate(self, data=None):
            self.stdin_data = data
            await asyncio.sleep(60)

        async def wait(self):
            self.waited = True
            return self.returncode

    proc = _HungProc()
    _patch_cli(monkeypatch, proc=proc)
    provider = build_provider(_ep())
    provider.timeout = 0.001

    with pytest.raises(ProviderError, match="timed out"):
        asyncio.run(provider.complete([user("hang")]))
    assert proc.killed is True
    assert proc.waited is True


# ---- local non-secret model catalog -----------------------------------------


def test_discover_codex_models_reads_visible_slugs(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.6-sol", "visibility": "list"},
                    {"slug": "gpt-5.6-luna"},
                    {"slug": "internal-hidden", "visibility": "hide"},
                    {"visibility": "list"},
                    {"slug": "gpt-5.6-sol", "visibility": "list"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert codex.discover_codex_models() == ["gpt-5.6-luna", "gpt-5.6-sol"]


def test_discover_codex_models_degrades_on_missing_or_invalid_cache(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert codex.discover_codex_models() == []
    (tmp_path / "models_cache.json").write_text("{broken", encoding="utf-8")
    assert codex.discover_codex_models() == []


def test_codex_login_status_uses_cli_without_secret_env(monkeypatch):
    captured = {}

    def run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT\n", stderr="")

    monkeypatch.setenv("WALLBREAKER_CODEX_BIN", "/opt/test/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setattr(codex.subprocess, "run", run)

    ready, detail = codex.codex_login_status()

    assert ready is True
    assert detail == "Logged in using ChatGPT"
    assert captured["args"] == ["/opt/test/codex", "login", "status"]
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]


def test_codex_login_status_reports_missing_binary(monkeypatch):
    monkeypatch.delenv("WALLBREAKER_CODEX_BIN", raising=False)
    monkeypatch.setattr(codex.shutil, "which", lambda _name: None)
    assert codex.codex_login_status() == (False, "codex CLI not found")

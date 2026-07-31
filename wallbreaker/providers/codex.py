from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.messages import (
    Message,
    ReasoningDelta,
    StopEvent,
    StreamEvent,
    TextDelta,
    ToolUseEvent,
    UsageEvent,
)
from .base import DEFAULT_TIMEOUT, Provider, ProviderError
from .claude_code import _parse_tool_calls, _render_conversation, _render_tools

_MIN_TIMEOUT = 300.0
_RUNTIME_GUARD = (
    "\n\n# CODEX CLI ADAPTER BOUNDARY\n"
    "Do not invoke native Codex tools, inspect files or environment variables, run shell "
    "commands, use the network, or delegate to subagents. The working directory is "
    "intentionally empty. Respond directly from the transcript supplied on stdin. When "
    "HARNESS TOOLS are listed above, the literal <tool_call> text protocol is the only "
    "action mechanism available to you."
)
_SAFE_ENV_NAMES = {
    "ALL_PROXY",
    "CODEX_HOME",
    "COMSPEC",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "HOME",
    "LANG",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SystemRoot",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "__CF_USER_TEXT_ENCODING",
}


@dataclass
class _CodexOutput:
    text: str = ""
    reasoning: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    agent_seen: bool = False
    completed: bool = False
    failed: str = ""
    errors: list[str] = field(default_factory=list)


def _error_text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("message") or value.get("error") or value)
    return str(value or "")


def _parse_jsonl(raw: str) -> _CodexOutput:
    result = _CodexOutput()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"codex CLI returned malformed JSONL on line {line_number}: {line[:160]}"
            ) from exc
        if not isinstance(event, dict):
            raise ProviderError(
                f"codex CLI returned a non-object JSONL event on line {line_number}"
            )
        event_type = str(event.get("type") or "")
        if event_type == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            text = item.get("text")
            if item_type == "agent_message":
                result.agent_seen = True
                result.text = str(text or "")
            elif item_type == "reasoning" and text:
                result.reasoning.append(str(text))
            # item.type=error is a warning in Codex JSONL and may be followed by a
            # successful turn. Native Codex tool items are intentionally ignored.
        elif event_type == "turn.completed":
            result.completed = True
            result.usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        elif event_type == "turn.failed":
            result.failed = _error_text(event.get("error")) or "unknown error"
        elif event_type == "error":
            # Top-level errors may be retryable. Keep the diagnostic and decide only
            # after the process exits and a terminal event is known.
            result.errors.append(_error_text(event.get("message") or event.get("error")))
    return result


def _safe_subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENV_NAMES or key.startswith("LC_")
    }
    return env


def discover_codex_models() -> list[str]:
    """Read Codex's non-secret local model catalog without touching its auth state."""
    codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    try:
        payload = json.loads((codex_home / "models_cache.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    models = {
        str(row.get("slug")).strip()
        for row in rows
        if isinstance(row, dict)
        and row.get("slug")
        and str(row.get("visibility") or "list") != "hide"
    }
    return sorted(models, key=str.casefold)


def codex_login_status(timeout: float = 10.0) -> tuple[bool, str]:
    """Check the local CLI/login without reading or copying its credential files."""
    binary = os.environ.get("WALLBREAKER_CODEX_BIN") or shutil.which("codex")
    if not binary:
        return False, "codex CLI not found"
    try:
        result = subprocess.run(
            [binary, "login", "status"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"codex login status failed: {exc}"
    output_lines = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    login_line = next(
        (line for line in output_lines if "logged in" in line.casefold()),
        "",
    )
    message = login_line or result.stdout.strip()
    if result.returncode == 0:
        return True, message or "Codex login is active"
    detail = message or result.stderr.strip()
    return False, detail[:300] or f"codex login status exited {result.returncode}"


class CodexProvider(Provider):
    """Run Codex CLI through the user's existing ChatGPT/Codex login.

    Codex is an agent surface rather than a raw chat-completions endpoint. Each harness
    call is therefore an isolated, ephemeral `codex exec` invocation. Native tools and
    external context are disabled; Wallbreaker tools use the same explicit text protocol
    as the Claude Code adapter.
    """

    supports_native_prefill = False

    def __init__(self, endpoint, timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__(endpoint, timeout=max(timeout, _MIN_TIMEOUT))
        self.bin = os.environ.get("WALLBREAKER_CODEX_BIN") or shutil.which("codex") or "codex"
        self.last_stop_reason: str | None = None
        self.last_completion_empty = False

    def _args(self, workdir: str, system: str) -> list[str]:
        args = [
            self.bin,
            "-a",
            "never",
            "exec",
            "--json",
            "--color",
            "never",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            workdir,
            "--disable",
            "shell_tool",
            "--disable",
            "shell_snapshot",
            "--disable",
            "apps",
            "--disable",
            "multi_agent",
            "--disable",
            "memories",
            "-c",
            'web_search="disabled"',
            "-c",
            'tools.view_image=false',
            "-c",
            'project_doc_max_bytes=0',
            "-c",
            'allow_login_shell=false',
            "-c",
            'shell_environment_policy.inherit="none"',
            "-c",
            "developer_instructions=" + json.dumps(system, ensure_ascii=False),
        ]
        model = str(self.endpoint.model or "")
        if model:
            args += ["-m", model]
        effort = str(getattr(self.endpoint, "reasoning_effort", "") or "")
        if effort:
            args += ["-c", "model_reasoning_effort=" + json.dumps(effort)]
        return args + ["-"]

    async def _terminate(self, proc) -> None:
        if getattr(proc, "returncode", None) is not None:
            return
        try:
            if os.name == "posix" and getattr(proc, "pid", None):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        wait = getattr(proc, "wait", None)
        if wait is not None:
            try:
                await asyncio.wait_for(wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

    async def _run_cli(self, prompt: str, system: str) -> _CodexOutput:
        with tempfile.TemporaryDirectory(prefix="wallbreaker-codex-") as workdir:
            args = self._args(workdir, system)
            kwargs = {
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "cwd": workdir,
                "env": _safe_subprocess_env(),
            }
            if os.name == "posix":
                kwargs["start_new_session"] = True
            try:
                proc = await asyncio.create_subprocess_exec(*args, **kwargs)
            except FileNotFoundError as exc:
                raise ProviderError(
                    f"codex CLI not found (looked for '{self.bin}'). Install Codex CLI or "
                    "set WALLBREAKER_CODEX_BIN to its path."
                ) from exc
            communicate = asyncio.create_task(proc.communicate(prompt.encode("utf-8")))
            try:
                stdout, stderr = await asyncio.wait_for(communicate, timeout=self.timeout)
            except asyncio.CancelledError:
                await self._terminate(proc)
                if not communicate.done():
                    communicate.cancel()
                await asyncio.gather(communicate, return_exceptions=True)
                raise
            except (asyncio.TimeoutError, TimeoutError) as exc:
                await self._terminate(proc)
                if not communicate.done():
                    communicate.cancel()
                await asyncio.gather(communicate, return_exceptions=True)
                raise ProviderError(
                    f"codex CLI timed out after {int(self.timeout)}s"
                ) from exc
            except Exception as exc:
                await self._terminate(proc)
                raise ProviderError(f"codex CLI process failed: {exc}") from exc

            raw = (stdout or b"").decode("utf-8", "replace")
            err = (stderr or b"").decode("utf-8", "replace").strip()[:500]
            parsed = _parse_jsonl(raw)
            if parsed.failed:
                raise ProviderError(f"codex CLI turn failed: {parsed.failed}")
            if proc.returncode != 0:
                detail = next((item for item in reversed(parsed.errors) if item), "") or err
                suffix = f": {detail}" if detail else ""
                raise ProviderError(f"codex CLI exited {proc.returncode}{suffix}")
            if not parsed.completed:
                detail = next((item for item in reversed(parsed.errors) if item), "") or err
                suffix = f": {detail}" if detail else ""
                raise ProviderError(f"codex CLI exited without turn.completed{suffix}")
            if not parsed.agent_seen:
                raise ProviderError("codex CLI completed without an agent message")
            return parsed

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del max_tokens, temperature  # codex exec does not expose these completion controls
        full_system = system or ""
        if tools:
            full_system += _render_tools(tools)
        full_system += _RUNTIME_GUARD
        prompt = _render_conversation(messages) or (messages[-1].text() if messages else "")
        data = await self._run_cli(prompt, full_system)

        residual, calls = _parse_tool_calls(data.text) if tools else (data.text, [])
        self.last_stop_reason = "tool_use" if calls else "end_turn"
        self.last_completion_empty = not residual and not calls

        for item in data.reasoning:
            yield ReasoningDelta(item)
        if residual:
            yield TextDelta(residual)
        for index, call in enumerate(calls):
            yield ToolUseEvent(id=f"codex_{index}", name=call.name, input=call.input)

        usage = data.usage
        yield UsageEvent(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cached_input_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_write_input_tokens", 0) or 0),
        )
        yield StopEvent(self.last_stop_reason)

from __future__ import annotations

import asyncio
import functools
import posixpath
from dataclasses import dataclass, replace
from importlib import resources
from typing import TYPE_CHECKING, Any, TypeGuard

from deepagents.backends.protocol import RAW_FILE_RESULT_CONFIG_KEY
from langchain_core.messages import ToolCall, ToolMessage

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Awaitable, Callable, Collection, Mapping

    from deepagents.backends.protocol import (
        FileToolErrorArtifact,
        LsArtifact,
        RawFileResultRequest,
        ReadFileArtifact,
    )
    from langchain_core.tools import BaseTool
    from langgraph.errors import GraphInterrupt
    from langgraph.prebuilt.tool_node import ToolCallRequest

HOST_FUNCTION = "__lc_fs"

APPROVAL = (
    "`{tool}` needs human approval, which a call from code cannot request. "
    "In your next turn, call `{tool}` directly, then rerun the rest of this code."
)

PAUSED = (
    "`{tool}` paused the run, which a call from code cannot do. "
    "In your next turn, call `{tool}` directly, then rerun the rest of this code."
)


@dataclass(frozen=True, slots=True)
class SyncChain:
    request: ToolCallRequest
    handler: Callable[[ToolCallRequest], Any]
    context: contextvars.Context
    """The eval call's context, where LangGraph keeps the run's config."""

    def submit(self, request: ToolCallRequest) -> Awaitable[Any]:
        context = self.context.copy()
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, context.run, self.handler, request)


@dataclass(frozen=True, slots=True)
class AsyncChain:
    request: ToolCallRequest
    handler: Callable[[ToolCallRequest], Any]
    loop: asyncio.AbstractEventLoop
    context: contextvars.Context
    """The eval call's context, where LangGraph keeps the run's config."""

    def submit(self, request: ToolCallRequest) -> Awaitable[Any]:
        context = self.context.copy()

        async def run() -> Any:
            return await asyncio.create_task(
                context.run(self.handler, request), context=context
            )

        return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(run(), self.loop))


ToolChain = SyncChain | AsyncChain


class FsError(Exception):
    def __init__(self, message: str, *, code: str, **props: str) -> None:
        super().__init__(message)
        self.code = code
        self.props = props

    @property
    def reply(self) -> dict[str, Any]:
        return {"error": {"message": str(self), "code": self.code, **self.props}}


def _strict_utf8(op: str, path: str, artifact: ReadFileArtifact) -> str:
    if path.endswith("/"):
        raise _system_error(op, path, "ENOTDIR")
    try:
        return artifact["content"].decode("utf-8")
    except UnicodeDecodeError:
        msg = "The encoded data was not valid for encoding utf-8"
        code = "ERR_ENCODING_INVALID_ENCODED_DATA"
        raise FsError(msg, code=code, name="TypeError") from None


def _names(_op: str, _path: str, artifact: LsArtifact) -> list[str]:
    return [posixpath.basename(e["path"].rstrip("/")) for e in artifact["entries"]]


def _is_directory(op: str, path: str, artifact: LsArtifact) -> bool:
    target = path.rstrip("/")
    if not target:
        return True
    entries = (e for e in artifact["entries"] if e["path"].rstrip("/") == target)
    entry = next(entries, None)
    if entry is None:
        raise _system_error(op, path, "ENOENT")
    if path != target and not entry["is_dir"]:
        raise _system_error(op, path, "ENOTDIR")
    return entry["is_dir"]


def _parent(path: str, _args: dict[str, Any]) -> dict[str, Any]:
    return {"path": posixpath.dirname(path)}


@dataclass(frozen=True, slots=True)
class _Operation:
    fs_function: str
    tool: str
    syscall: str
    arguments: Callable[[str, dict[str, Any]], dict[str, Any]]
    value: Callable[[str, str, Any], Any] | None
    """`None` for a call that resolves to `undefined` and needs no raw result."""


_OPERATIONS = {
    "readFile": _Operation(
        "readFile", "read_file", "open", lambda p, _: {"file_path": p}, _strict_utf8
    ),
    "writeFile": _Operation(
        "writeFile",
        "write_file",
        "open",
        lambda p, a: {"file_path": p, "content": a["data"]},
        None,
    ),
    "readdir": _Operation("readdir", "ls", "scandir", lambda p, _: {"path": p}, _names),
    "access": _Operation("access", "ls", "access", _parent, _is_directory),
    # Node's `rm` names `lstat` when the path is missing.
    "rmStat": _Operation("rm", "ls", "lstat", _parent, _is_directory),
    "rm": _Operation("rm", "delete", "rm", lambda p, _: {"file_path": p}, None),
}

TOOLS = frozenset(op.tool for op in _OPERATIONS.values())

_FUNCTIONS = (
    (
        {"read_file"},
        "- `await fs.readFile(path)` returns the whole file as a string, without "
        'line numbers or headers, as in `JSON.parse(await fs.readFile("/a.json"))`.',
    ),
    (
        {"write_file"},
        "- `await fs.writeFile(path, text)` creates or replaces a file, creating "
        "parent directories.",
    ),
    (
        {"ls"},
        "- `await fs.readdir(dir)` lists entry names. `await fs.access(path)` "
        "rejects with `ENOENT` if the path is missing; as a boolean, "
        "`await fs.access(path).then(() => true, () => false)`.",
    ),
    (
        {"delete", "ls"},
        "- `await fs.rm(path, { recursive, force })` deletes as Node's does: a "
        "directory needs `recursive: true`, and `force: true` ignores a missing path.",
    ),
)

_NODE_CODES: dict[str | None, str] = {
    "file_not_found": "ENOENT",
    "is_directory": "EISDIR",
    "permission_denied": "EACCES",
    "invalid_path": "EINVAL",
    "not_a_directory": "ENOTDIR",
}

_ERROR_TEXT = {
    "ENOENT": "no such file or directory",
    "EISDIR": "illegal operation on a directory",
    "ENOTDIR": "not a directory",
    "EACCES": "permission denied",
    "EINVAL": "invalid argument",
    "EIO": "i/o error",
    "ENOSYS": "function not implemented",
}


def docs(tool_names: Collection[str]) -> list[str]:
    return [line for needs, line in _FUNCTIONS if needs <= set(tool_names)]


def public_name(op: str) -> str:
    return f"fs.{_OPERATIONS[op].fs_function}"


def plan(
    op: str, args: dict[str, Any], tools: Mapping[str, BaseTool]
) -> tuple[str, BaseTool, dict[str, Any]]:
    operation = _OPERATIONS[op]
    target = "/" + posixpath.normpath("/" + args["path"]).lstrip("/")
    directory = args["path"].endswith("/") and target != "/"
    path = target + "/" if directory else target
    tool = tools.get(operation.tool)
    if tool is None:
        reason = f"this agent has no `{operation.tool}` tool"
        raise _system_error(op, path, "ENOSYS", reason)
    if directory and operation.tool == "write_file":
        raise _system_error(op, path, "EISDIR")
    return path, tool, operation.arguments(target, args)


def refused(op: str, path: str, reason: str) -> FsError:
    return _system_error(op, path, "EACCES", reason)


def failed(op: str, path: str, error: Exception) -> FsError:
    return _system_error(op, path, "EIO", str(error))


def interruption(tool: str, interrupt: GraphInterrupt) -> tuple[str, str]:
    values = [i.value for i in interrupt.args[0]]
    if any(isinstance(v, dict) and v.get("type") == "tool_approval" for v in values):
        return "ApprovalRequired", APPROVAL.format(tool=tool)
    return "Interrupted", PAUSED.format(tool=tool)


def nested_request(
    chain: ToolChain, tool: BaseTool, arguments: dict[str, Any], call_id: str
) -> ToolCallRequest:
    runtime = chain.request.runtime
    raw: RawFileResultRequest = {"tool_call_id": call_id, "tool_name": tool.name}
    configurable = runtime.config.get("configurable", {})
    configurable = {**configurable, RAW_FILE_RESULT_CONFIG_KEY: raw}
    config = {**runtime.config, "configurable": configurable}
    call = ToolCall(name=tool.name, args=arguments, id=call_id, type="tool_call")
    nested_runtime = replace(runtime, config=config, tool_call_id=call_id)
    return replace(chain.request, tool_call=call, tool=tool, runtime=nested_runtime)


def result(op: str, path: str, outcome: Any) -> Any:
    operation = _OPERATIONS[op]
    if not isinstance(outcome, ToolMessage):
        reason = f"`{operation.tool}` returned a `Command`, which fs cannot apply"
        raise _system_error(op, path, "EIO", reason)
    if outcome.status == "error":
        raise _failure(op, path, outcome)
    if operation.value is None:
        return None
    if outcome.artifact is None:
        reason = f"`{operation.tool}` returned no raw result"
        raise _system_error(op, path, "ENOSYS", reason)
    return operation.value(op, path, outcome.artifact)


def _failure(op: str, path: str, message: ToolMessage) -> FsError:
    text = message.text.removeprefix("Error: ")
    if not _is_tool_error(message.artifact):
        # The file tool attaches an error artifact to every raw-request error,
        # so a refusal without one came from middleware above the tool.
        return _system_error(op, path, "EACCES", text)
    code = _NODE_CODES.get(message.artifact["code"])
    if code is not None:
        return _system_error(op, path, code, text if code == "EACCES" else None)
    return _system_error(op, path, "EIO", text)


def _is_tool_error(artifact: object) -> TypeGuard[FileToolErrorArtifact]:
    return isinstance(artifact, dict) and "code" in artifact


def _system_error(op: str, path: str, code: str, reason: str | None = None) -> FsError:
    syscall = _OPERATIONS[op].syscall
    message = f"{code}: {_ERROR_TEXT[code]}, {syscall} '{path}'"
    if reason:
        message += f" ({reason})"
    return FsError(message, code=code, syscall=syscall, path=path)


@functools.cache
def install_source() -> str:
    shim = (
        resources.files("langchain_quickjs").joinpath("_fs_shim.js").read_text("utf-8")
    )
    return f"({shim})({HOST_FUNCTION}); undefined"

from __future__ import annotations

import html
import json
import re
from typing import TYPE_CHECKING, Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.messages import BaseMessage

FILES = {
    "/data.json": json.dumps({"rows": list(range(150))}, indent=1),
    "/docs/a.md": "hello\nworld",
    "/docs/b.txt": "plain",
}

CODE_OF = "const codeOf = (f) => f().then(() => 'ok', (e) => e.code);\n"


class RecordingModel(FakeChatModel):
    prompts: list[str] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return super()._generate(messages, *args, **kwargs)


def model(*codes: str, then: tuple[dict[str, Any], ...] = ()) -> RecordingModel:
    evals = [{"name": "eval", "args": {"code": c}} for c in codes]
    script = [
        AIMessage("", tool_calls=[{**call, "id": f"c{i}"}])
        for i, call in enumerate([*evals, *then])
    ]
    return RecordingModel(messages=iter([*script, AIMessage(content="done")]))


def results(state: dict[str, Any]) -> list[Any]:
    out = []
    for message in state["messages"]:
        if message.type != "tool" or message.name != "eval":
            continue
        match = re.fullmatch(r"<result>(.*)</result>", message.content, re.DOTALL)
        out.append(json.loads(html.unescape(match[1])) if match else message.content)
    return out


def files() -> dict[str, Any]:
    return {path: create_file_data(text) for path, text in FILES.items()}


def run(code: str, **agent_kwargs: Any) -> Any:
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware()], **agent_kwargs
    )
    state = agent.invoke(
        {"messages": [{"role": "user", "content": "go"}], "files": files()}
    )
    return results(state)[-1]


def test_reads_files_as_text() -> None:
    code = """JSON.stringify({
      read: await fs.readFile("/docs/a.md"),
      rows: JSON.parse(await fs.promises.readFile("/data.json", "utf8")).rows.length,
    })"""

    assert run(code) == {"read": "hello\nworld", "rows": 150}


def test_writes_lists_and_deletes_in_one_eval() -> None:
    code = """const out = {};
    await fs.writeFile("/w/new.txt", "made");
    out.readBack = await fs.readFile("/w/new.txt");
    out.listed = (await fs.readdir("/docs")).sort();
    out.exists = await fs.access("/docs/a.md").then(() => true, () => false);
    out.root = await fs.access("/").then(() => true, () => false);
    out.mode = await fs.access("/docs/a.md", fs.constants.R_OK).then(
      () => [fs.constants.F_OK, fs.promises.constants.W_OK],
    );
    await fs.rm("/docs/b.txt");
    out.deleted = await fs.access("/docs/b.txt").then(() => "kept", (e) => e.code);
    JSON.stringify(out)"""
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware()]
    )

    state = agent.invoke(
        {"messages": [{"role": "user", "content": "go"}], "files": files()}
    )

    assert results(state) == [
        {
            "readBack": "made",
            "listed": ["a.md", "b.txt"],
            "exists": True,
            "root": True,
            "mode": [0, 2],
            "deleted": "ENOENT",
        }
    ]
    assert state["files"]["/w/new.txt"]["content"] == "made"
    assert "/docs/b.txt" not in state["files"]


def test_failures_are_node_errors(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("hello")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("hunter2")
    (tmp_path / "latin1.txt").write_bytes("café".encode("latin-1"))
    code = (
        CODE_OF
        + """JSON.stringify({
      missing: await codeOf(() => fs.readFile("/missing.txt")),
      readDirectory: await codeOf(() => fs.readFile("/docs")),
      deniedRead: await codeOf(() => fs.readFile("/secret/key.txt")),
      listFile: await codeOf(() => fs.readdir("/docs/a.md")),
      underFile: await codeOf(() => fs.access("/docs/a.md/x")),
      deniedListing: await codeOf(() => fs.access("/secret/inner/key.txt")),
      deniedFileHiddenFromListing: await codeOf(() => fs.access("/secret/key.txt")),
      writeDirectory: await codeOf(() => fs.writeFile("/docs", "x")),
      writeDirectoryPath: await codeOf(() => fs.writeFile("/new/", "x")),
      accessFileAsDirectory: await codeOf(() => fs.access("/docs/a.md/")),
      readFileAsDirectory: await codeOf(() => fs.readFile("/docs/a.md/")),
      notUtf8: await codeOf(() => fs.readFile("/latin1.txt")),
      rmMissing: await codeOf(() => fs.rm("/missing.txt")),
      rmMissingForced: await codeOf(() => fs.rm("/missing.txt", { force: true })),
      rmDirectory: await codeOf(() => fs.rm("/docs")),
      rmDirectoryRecursive: await codeOf(() => fs.rm("/docs", { recursive: true })),
    })"""
    )
    deny = FilesystemPermission(operations=["read"], paths=["/secret/**"], mode="deny")

    result = run(
        code,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        permissions=[deny],
    )

    assert result == {
        "missing": "ENOENT",
        "readDirectory": "EISDIR",
        "deniedRead": "EACCES",
        "listFile": "ENOTDIR",
        "underFile": "ENOTDIR",
        "deniedListing": "EACCES",
        "deniedFileHiddenFromListing": "ENOENT",
        "writeDirectory": "EISDIR",
        "writeDirectoryPath": "EISDIR",
        "accessFileAsDirectory": "ENOTDIR",
        "readFileAsDirectory": "ENOTDIR",
        "notUtf8": "ERR_ENCODING_INVALID_ENCODED_DATA",
        "rmMissing": "ENOENT",
        "rmMissingForced": "ok",
        "rmDirectory": "ERR_FS_EISDIR",
        "rmDirectoryRecursive": "ok",
    }
    assert (tmp_path / "latin1.txt").read_bytes() == "café".encode("latin-1")
    assert not (tmp_path / "docs").exists()
    assert not (tmp_path / "new").exists()


def test_state_backend_failures_are_node_errors() -> None:
    code = (
        CODE_OF
        + """const syscall = (f) => f().catch((e) => e.syscall);
    JSON.stringify({
      readMissing: await codeOf(() => fs.readFile("/missing.txt")),
      accessMissing: await codeOf(() => fs.access("/missing.txt")),
      accessFileAsDirectory: await codeOf(() => fs.access("/docs/a.md/")),
      writeDirectoryPath: await codeOf(() => fs.writeFile("/docs/", "x")),
      rmMissing: await codeOf(() => fs.rm("/missing.txt")),
      rmDirectory: await codeOf(() => fs.rm("/docs")),
      syscalls: [
        await syscall(() => fs.access("/missing.txt")),
        await syscall(() => fs.rm("/missing.txt")),
      ],
    })"""
    )

    assert run(code) == {
        "readMissing": "ENOENT",
        "accessMissing": "ENOENT",
        "accessFileAsDirectory": "ENOTDIR",
        "writeDirectoryPath": "EISDIR",
        "rmMissing": "ENOENT",
        "rmDirectory": "ERR_FS_EISDIR",
        "syscalls": ["access", "lstat"],
    }


def test_arguments_fs_cannot_honor_are_refused() -> None:
    code = (
        CODE_OF
        + """JSON.stringify({
      base64: await codeOf(() => fs.readFile("/docs/b.txt", "base64")),
      bytes: await codeOf(() => fs.writeFile("/w.txt", new Uint8Array([1]))),
      append: await codeOf(() => fs.writeFile("/w.txt", "x", { flag: "a" })),
      replace: await codeOf(() => fs.writeFile("/w.txt", "x", { flag: "w" })),
      fileTypes: await codeOf(() => fs.readdir("/docs", { withFileTypes: true })),
      recursive: await codeOf(() => fs.readdir("/docs", { recursive: true })),
      buffer: await codeOf(() => fs.readdir("/docs", "buffer")),
      callback: await codeOf(() => fs.readFile("/docs/b.txt", "utf8", () => {})),
      notAPath: await codeOf(() => fs.readFile(42)),
    })"""
    )

    assert run(code) == {
        "base64": "ERR_INVALID_ARG_VALUE",
        "bytes": "ERR_INVALID_ARG_TYPE",
        "append": "ERR_INVALID_ARG_VALUE",
        "replace": "ok",
        "fileTypes": "ERR_INVALID_ARG_VALUE",
        "recursive": "ERR_INVALID_ARG_VALUE",
        "buffer": "ERR_INVALID_ARG_VALUE",
        "callback": "ERR_INVALID_ARG_TYPE",
        "notAPath": "ERR_INVALID_ARG_TYPE",
    }


def test_sync_and_dropped_functions_throw_an_error_naming_the_replacement() -> None:
    code = """const thrown = (f) => {
      try { f(); return null } catch (e) { return [e.code, e.message] }
    };
    const dropped = `readFileSync writeFileSync readdirSync existsSync exists
      accessSync stat statSync lstat glob rmSync unlink rmdir appendFile mkdir`;
    JSON.stringify({
      dropped: Object.fromEntries(dropped.split(/\\s+/).map(
        (name) => [name, thrown(() => fs.promises[name]("/docs/a.md"))],
      )),
      require: thrown(() => require("fs"))[0],
      promisesReadFile: typeof fs.promises.readFile,
    })"""

    result = run(code)

    dropped = result.pop("dropped")
    assert len(dropped) == 15
    assert {code for code, _ in dropped.values()} == {"ERR_METHOD_NOT_IMPLEMENTED"}
    assert "Use await fs.readFile(path)." in dropped["readFileSync"][1]
    assert "Use await fs.access(path)," in dropped["existsSync"][1]
    assert result == {"require": "MODULE_NOT_FOUND", "promisesReadFile": "function"}

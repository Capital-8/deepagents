from __future__ import annotations

import asyncio
import html
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests.unit_tests.test_fs import CODE_OF, files, model, results

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.tools.tool_node import ToolCallRequest

_CONFIG = {"configurable": {"thread_id": "t"}}
_GO = {"messages": [{"role": "user", "content": "go"}]}


def _invoke(agent: Any, payload: Any, *, run_async: bool = False) -> dict[str, Any]:
    if run_async:
        return asyncio.run(agent.ainvoke(payload, _CONFIG))
    return agent.invoke(payload, _CONFIG)


def _eval_message(state: dict[str, Any]) -> str:
    return next(m.content for m in state["messages"] if m.name == "eval")


class _Recorder(AgentMiddleware[Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self.calls.append(request.tool_call["name"])
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self.calls.append(request.tool_call["name"])
        return await handler(request)


class _RunsToolsItself(AgentMiddleware[Any, Any, Any]):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        del handler
        args = {**request.tool_call["args"], "runtime": request.runtime}
        return request.tool.invoke({**request.tool_call, "args": args})


class _PauseReads(AgentMiddleware[Any, Any, Any]):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] == "read_file":
            interrupt("continue?")
        return handler(request)


class _RefuseReads(AgentMiddleware[Any, Any, Any]):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] != "read_file":
            return handler(request)
        return ToolMessage(
            "Error: reads need review",
            name="read_file",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


class _Hide(AgentMiddleware[Any, Any, Any]):
    def __init__(self, tool: str, *, after: int = 0) -> None:
        super().__init__()
        self.tool = tool
        self.after = after
        self.calls = 0

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self.calls += 1
        if self.calls <= self.after:
            return handler(request)
        tools = [t for t in request.tools if getattr(t, "name", "") != self.tool]
        return handler(request.override(tools=tools))


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_middleware_after_the_interpreter_sees_each_fs_call(*, run_async: bool) -> None:
    code = """const [a, b] = await Promise.all([
      fs.readFile("/docs/a.md"), fs.readFile("/docs/b.txt"),
    ]);
    await fs.writeFile("/w.txt", a + b);
    JSON.stringify([a, b])"""
    recorder = _Recorder()
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware(), recorder]
    )

    state = _invoke(agent, {**_GO, "files": files()}, run_async=run_async)

    assert results(state) == [["hello\nworld", "plain"]]
    assert sorted(recorder.calls) == ["eval", "read_file", "read_file", "write_file"]
    assert state["files"]["/w.txt"]["content"] == "hello\nworldplain"


def test_a_refusal_from_middleware_throws_eacces_with_its_reason() -> None:
    code = """let error;
    try { await fs.readFile("/docs/b.txt") }
    catch (e) { error = [e.code, e.message] }
    JSON.stringify(error)"""
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware(), _RefuseReads()]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert results(state) == [
        ["EACCES", "EACCES: permission denied, open '/docs/b.txt' (reads need review)"]
    ]


@pytest.mark.parametrize(
    ("call", "tool"),
    [('fs.writeFile("/a.txt", "y")', "write_file"), ('fs.rm("/a.txt")', "delete")],
)
def test_a_change_gated_per_call_ends_the_eval_and_changes_nothing(
    tmp_path: Path, call: str, tool: str
) -> None:
    (tmp_path / "a.txt").write_text("x")
    code = f"try {{ await {call} }} catch (e) {{ console.log(e.code) }}"
    approval = HumanInTheLoopMiddleware(
        interrupt_on={tool: True}, interrupt_mode="per_call"
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    agent = create_agent(
        model(code),
        middleware=[
            FilesystemMiddleware(backend=backend),
            CodeInterpreterMiddleware(),
            approval,
        ],
        checkpointer=InMemorySaver(),
    )

    state = _invoke(agent, _GO)

    assert _eval_message(state).startswith(
        f'<stdout>\nEACCES\n</stdout>\n<error type="ApprovalRequired">`{tool}`'
    )
    assert (tmp_path / "a.txt").read_text() == "x"


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_per_call_approval_ends_the_eval_and_still_reviews_the_model(
    tmp_path: Path, *, run_async: bool
) -> None:
    (tmp_path / "notes.md").write_text("secret notes")
    (tmp_path / "other.md").write_text("open notes")
    ran: list[str] = []

    @tool
    def count() -> str:
        """Count a call."""
        ran.append("count")
        return "counted"

    code = """let first, later;
    try { await fs.readFile("/notes.md") } catch (e) { first = e.code }
    try { await fs.readFile("/other.md") } catch (e) { later = [e.code, e.message] }
    const counted = await tools.count({}).then(() => "ran", () => "rejected");
    console.log(JSON.stringify([first, ...later, counted]));
    "finished\""""
    read = {"name": "read_file", "args": {"file_path": "/notes.md"}}
    gate = {
        "allowed_decisions": ["approve", "reject"],
        "when": lambda request: request.tool_call["args"]["file_path"] == "/notes.md",
    }
    approval = HumanInTheLoopMiddleware(
        interrupt_on={"read_file": gate}, interrupt_mode="per_call"
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    agent = create_agent(
        model(code, then=(read,)),
        tools=[count],
        middleware=[
            FilesystemMiddleware(backend=backend),
            CodeInterpreterMiddleware(ptc=["count"]),
            approval,
        ],
        checkpointer=InMemorySaver(),
    )

    paused = _invoke(agent, _GO, run_async=run_async)
    resumed = _invoke(agent, Command(resume={"type": "approve"}), run_async=run_async)

    advice = (
        "`read_file` needs human approval, which a call from code cannot request. In "
        "your next turn, call `read_file` directly, then rerun the rest of this code."
    )
    later = f"EACCES: permission denied, open '/other.md' ({advice})"
    assert html.unescape(_eval_message(paused)) == (
        f'<stdout>\n["EACCES","EACCES","{later}","rejected"]\n</stdout>\n'
        f'<error type="ApprovalRequired">{advice}</error>'
    )
    assert ran == []
    assert "__interrupt__" in paused
    assert "__interrupt__" not in resumed
    assert "secret notes" in resumed["messages"][-2].content


def test_a_pause_that_is_not_an_approval_ends_the_eval_as_interrupted() -> None:
    agent = create_deep_agent(
        model=model('await fs.readFile("/docs/b.txt")'),
        middleware=[CodeInterpreterMiddleware(), _PauseReads()],
        checkpointer=InMemorySaver(),
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert _eval_message(state) == (
        '<error type="Interrupted">`read_file` paused the run, which a call from '
        "code cannot do. In your next turn, call `read_file` directly, then rerun "
        "the rest of this code.</error>"
    )


@tool
def ask(question: str) -> str:
    """Ask the user a question."""
    return str(interrupt(question))


def _asking_agent(tmp_path: Path, code: str) -> Any:
    (tmp_path / "notes.md").write_text("secret notes")
    approval = HumanInTheLoopMiddleware(
        interrupt_on={"read_file": True}, interrupt_mode="per_call"
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    return create_agent(
        model(code),
        tools=[ask],
        middleware=[
            FilesystemMiddleware(backend=backend),
            CodeInterpreterMiddleware(ptc=["ask"]),
            approval,
        ],
        checkpointer=InMemorySaver(),
    )


def test_a_resumed_eval_keeps_its_answer_and_ends_at_a_gated_fs_call(
    tmp_path: Path,
) -> None:
    code = """const answer = await tools.ask({ question: "color?" });
    console.log(answer);
    await fs.readFile("/notes.md");
    "unreached\""""
    agent = _asking_agent(tmp_path, code)

    paused = _invoke(agent, _GO)
    resumed = _invoke(agent, Command(resume="blue"))

    assert paused["__interrupt__"][0].value == "color?"
    assert "__interrupt__" not in resumed
    assert _eval_message(resumed).startswith(
        '<stdout>\nblue\n</stdout>\n<error type="ApprovalRequired">'
    )


def test_without_the_tool_call_middleware_fs_throws_eacces() -> None:
    code = CODE_OF + 'JSON.stringify(await codeOf(() => fs.readFile("/docs/b.txt")))'
    agent = create_deep_agent(
        model=model(code), middleware=[_RunsToolsItself(), CodeInterpreterMiddleware()]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert results(state) == ["EACCES"]


def test_an_oversized_read_ends_the_eval_with_out_of_memory(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("a" * (16 * 1024 * 1024))
    agent = create_deep_agent(
        model=model('(await fs.readFile("/big.txt")).length', "1 + 1"),
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        middleware=[CodeInterpreterMiddleware(memory_limit=8 * 1024 * 1024)],
    )

    state = _invoke(agent, _GO)

    assert 'type="OutOfMemory"' in results(state)[0]
    assert results(state)[1] == 2


@pytest.mark.parametrize("enabled", [True, False], ids=["default", "fs=False"])
def test_fs_is_on_by_default_and_off_with_the_opt_out(*, enabled: bool) -> None:
    llm = model("JSON.stringify([typeof fs, typeof require])")
    agent = create_deep_agent(
        model=llm, middleware=[CodeInterpreterMiddleware(fs=enabled)]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    expected = ["object", "function"] if enabled else ["undefined", "undefined"]
    assert results(state) == [expected]
    assert ("### Files: `fs`" in llm.prompts[0]) is enabled


def test_a_provider_tool_given_as_a_dict_leaves_fs_working() -> None:
    agent = create_deep_agent(
        model=model('(await fs.readFile("/docs/b.txt")).length'),
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        middleware=[CodeInterpreterMiddleware()],
    )

    assert isinstance(results(_invoke(agent, {**_GO, "files": files()}))[0], int)


def test_fs_is_undefined_when_no_file_tool_is_bound() -> None:
    llm = model("JSON.stringify(typeof fs)")
    agent = create_agent(llm, middleware=[CodeInterpreterMiddleware()])

    state = _invoke(agent, _GO)

    assert results(state) == ["undefined"]
    assert "### Files" not in llm.prompts[0]


def test_fs_throws_enosys_for_a_file_tool_the_model_cannot_see() -> None:
    code = (
        CODE_OF
        + """JSON.stringify([
      await codeOf(() => fs.writeFile("/w.txt", "x")),
      await fs.readFile("/docs/b.txt"),
    ])"""
    )
    llm = model(code)
    agent = create_deep_agent(
        model=llm, middleware=[_Hide("write_file"), CodeInterpreterMiddleware()]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert results(state) == [["ENOSYS", "plain"]]
    assert "/w.txt" not in state["files"]
    assert "fs.writeFile" not in llm.prompts[0]


def test_fs_throws_enosys_once_a_later_call_hides_read_file() -> None:
    code = """JSON.stringify([
      typeof fs, await fs.readFile("/docs/b.txt").catch((e) => e.code),
    ])"""
    agent = create_deep_agent(
        model=model(code, code),
        middleware=[_Hide("read_file", after=1), CodeInterpreterMiddleware()],
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert results(state) == [["object", "plain"], ["object", "ENOSYS"]]


def test_fs_calls_count_toward_the_call_budget() -> None:
    code = """for (const p of ["/docs/a.md", "/docs/b.txt", "/data.json"]) {
      await fs.readFile(p);
    }"""
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware(max_ptc_calls=2)]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert _eval_message(state) == (
        '<error type="PTCCallBudgetExceeded">PTC call budget exceeded '
        "(limit=2, attempted=3, function=fs.readFile)</error>"
    )


@pytest.mark.parametrize("function", ["access", "rm"])
def test_a_budget_error_names_the_fs_function_the_code_called(function: str) -> None:
    code = f'await fs.readFile("/docs/a.md"); await fs.{function}("/docs/b.txt")'
    agent = create_deep_agent(
        model=model(code), middleware=[CodeInterpreterMiddleware(max_ptc_calls=1)]
    )

    state = _invoke(agent, {**_GO, "files": files()})

    assert f"attempted=2, function=fs.{function})" in _eval_message(state)


def test_raw_read_results_stay_out_of_streams_and_checkpoints_of_a_turn_repl(
    tmp_path: Path,
) -> None:
    (tmp_path / "f.txt").write_text("MARKER" * 3)
    saver = InMemorySaver()
    llm = model('(await fs.readFile("/f.txt")).length')
    llm.disable_streaming = True  # the fake model cannot stream tool calls
    agent = create_deep_agent(
        model=llm,
        middleware=[CodeInterpreterMiddleware(mode="turn")],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        checkpointer=saver,
    )

    chunks = list(
        agent.stream(_GO, _CONFIG, stream_mode=["updates", "messages", "values"])
    )

    assert "<result>18</result>" in repr(chunks)
    assert "MARKER" not in repr(chunks)
    assert "MARKER" not in repr(list(saver.list(_CONFIG)))

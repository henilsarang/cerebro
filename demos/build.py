"""Build demo - a coding agent with real tools behind a security policy.

Concept: the loop is unchanged; this demo only fills its sockets. The tools
come from core_tools() fenced into a scratch directory, and Policy.check is
plugged in as before_tool, so dangerous commands come back as BLOCKED
results the model must respond to.

Design rules:
  * Each run gets a fresh scratch directory, never the repository itself.
  * yolo mode is safe enough for a demo only because the deny patterns
    still apply and every path is fenced to the scratch directory.

Run from the repo root:  python demos/build.py "your task"
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cerebro import provider  # noqa: E402
from cerebro.loop import run_loop  # noqa: E402
from cerebro.security import Policy  # noqa: E402
from cerebro.tools import core_tools  # noqa: E402

TASK = ("Create fib.py with an iterative fib(n), a main printing fib(30), "
        "run it and confirm the output is 832040")
SYSTEM = (
    "You are a careful coding agent working in {workdir}. Use the tools to "
    "read, write and run code, and verify claims by running them before you "
    "answer. If a tool result says BLOCKED or ERROR, do not try to get "
    "around it; explain briefly and politely what you could not do."
)


def on_event(kind, payload):
    """Print each loop event as one transcript line."""
    if kind == "assistant":
        for call in payload["tool_calls"]:
            print(f"assistant -> {call['name']}({call['args']})")
        if payload["text"]:
            print(f"assistant: {payload['text']}")
    elif kind == "tool_end":
        result = payload["result"]
        print(f"tool {payload['name']} -> {result[:300]}{'...' if len(result) > 300 else ''}")


def main(task=TASK, workdir=None):
    """Run task in a scratch directory under yolo policy; return the answer."""
    workdir = workdir or tempfile.mkdtemp(prefix="cerebro-")
    tools = {t.name: t for t in core_tools(workdir)}
    policy = Policy("yolo")
    print(f"workdir: {workdir}\nuser: {task}")
    messages = [{"role": "user", "text": task}]
    return run_loop(provider.DEFAULT_MODEL, SYSTEM.format(workdir=workdir),
                    messages, tools, on_event, before_tool=policy.check)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or TASK)

"""Day 1 demo - one hand-written tool driven through the agent loop.

Concept: a tool is just an object with a JSON-schema .spec the model reads
and a .run callable the harness executes. Here the model rolls dice, reads
the result, and reasons about it in plain text.

Design rules:
  * The tool is written by hand, with no registry or decorator, so every
    moving part is visible. Day 2 replaces this with a proper tool module.
  * Arguments arrive as the model sent them; count is declared as a string
    and converted here, the tool's job rather than the loop's.

Run from the repo root:  python demos/day1_dice.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cerebro import provider  # noqa: E402
from cerebro.loop import run_loop  # noqa: E402

TASK = "Roll 3 dice and tell me whether the total beats 10"


class RollDice:
    """Roll some six-sided dice and return the individual results."""

    spec = {"schema": {
        "name": "roll_dice",
        "description": "Roll count six-sided dice",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "string", "description": "How many dice"}},
            "required": ["count"],
        },
    }}

    def run(self, count):
        """Return a list of int(count) rolls, each between 1 and 6."""
        return [random.randint(1, 6) for _ in range(int(count))]


def on_event(kind, payload):
    """Print each loop event as one transcript line."""
    if kind == "assistant":
        for call in payload["tool_calls"]:
            print(f"assistant -> tool call {call['name']}({call['args']})")
        if payload["text"]:
            print(f"assistant: {payload['text']}")
    elif kind == "tool_end":
        print(f"tool {payload['name']} -> {payload['result']}")


def allow_all(call):
    """Permit every tool call (day 2 adds real policy)."""
    return None


def main(task=TASK):
    """Run the task through the loop and return the final answer."""
    print(f"user: {task}")
    messages = [{"role": "user", "text": task}]
    return run_loop(provider.DEFAULT_MODEL, "You are a helpful assistant.",
                    messages, {"roll_dice": RollDice()}, on_event, allow_all)


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or TASK)

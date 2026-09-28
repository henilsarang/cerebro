"""Day 1 - The agent loop: call the model, run its tools, repeat.

Concept: an agent is a loop. The model replies; if the reply asks for tools,
the harness runs them, appends the results, and asks again. When a reply
arrives with no tool calls, that text is the answer.

Design rules:
  * The loop never crashes because a tool did: every failure becomes a
    string result the model can read and recover from.
  * Policy lives outside the loop. before_tool decides what may run;
    on_event decides what gets shown; before_turn (day 3) reshapes context.
  * Bounded: after max_turns the model is told to wrap up and gets one
    final call with no tools, so it must answer in text.
"""
from . import provider


def _execute(call, tools, before_tool):
    """Run one tool call and return its result as a string, never raising."""
    reason = before_tool(call)
    if reason is not None:
        return f"BLOCKED: {reason}"
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"
    try:
        return str(tool.run(**call["args"]))
    except Exception as err:  # the model sees the failure; the loop survives
        return f"ERROR: {type(err).__name__}: {err}"


def run_loop(model, system, messages, tools, on_event, before_tool,
             max_turns=80, before_turn=None):
    """Drive the model until it answers without tool calls; return that text.

    messages is mutated in place so the caller keeps the full transcript.
    tools maps name -> Tool (with .spec and .run). on_event(kind, payload)
    fires "assistant", "tool_start" and "tool_end". before_tool(call) returns
    None to allow a call or a reason string to block it.
    """
    specs = [t.spec for t in tools.values()]
    for _ in range(max_turns):
        if before_turn is not None:
            messages[:] = before_turn(messages)
        reply = provider.complete(model, system, messages, specs)
        messages.append({"role": "assistant", "text": reply["text"],
                         "tool_calls": reply["tool_calls"]})
        on_event("assistant", reply)
        if not reply["tool_calls"]:
            return reply["text"]
        # Results are appended in call order so each lines up with its request.
        for call in reply["tool_calls"]:
            on_event("tool_start", call)
            result = _execute(call, tools, before_tool)
            on_event("tool_end", {"name": call["name"], "result": result})
            messages.append({"role": "tool", "name": call["name"], "text": result})

    messages.append({"role": "user", "text": "Turn limit reached; wrap up now."})
    reply = provider.complete(model, system, messages, [])
    messages.append({"role": "assistant", "text": reply["text"],
                     "tool_calls": reply["tool_calls"]})
    on_event("assistant", reply)
    return reply["text"]

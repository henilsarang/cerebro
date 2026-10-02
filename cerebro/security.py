"""Day 2 - Security: a policy that decides which tool calls may run.

Concept: the loop asks before_tool(call) before every execution. Policy.check
is that hook: it returns None to allow a call or a reason string to block it,
and the model reads "BLOCKED: <reason>" as the tool result.

Design rules:
  * Deny patterns win over everything, yolo mode included: some commands
    are never worth running unattended.
  * Reading is always safe; anything that changes the world depends on mode:
    "read-only" refuses it, "safe" asks a human, "yolo" allows it.
  * Fail closed. With no approver, safe mode refuses.
  * The deny list is a seatbelt, not a sandbox: it catches obvious disasters,
    while the working-directory fence in tools.py contains file access.
"""
import re

READ_TOOLS = {"read_file", "list_files", "grep"}

DENY_PATTERNS = [
    # rm with a recursive flag aimed at /, /*, ~ or $HOME itself.
    r"\brm\s+(-\w+\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(/|~|\$HOME|\$\{HOME\})/?\*?(\s|;|&|$)",
    r"\brm\s+(-\w+\s+)*--recursive\s+(-\w+\s+)*(/|~|\$HOME|\$\{HOME\})/?\*?(\s|;|&|$)",
    r"\bsudo\b",
    r"\bmkfs\b|\bdd\s+if=",
    r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b",
    r"\bgit\s+push\b.*(--force\b|\s-f\b)",
    r">\s*/dev/sd[a-z]",
]
MODES = ("read-only", "safe", "yolo")


def _refuse(call, reason):
    """Default approver: no human is listening, so the answer is no."""
    return False


class Policy:
    """Tool-call policy for one session: a mode plus an optional approver."""

    def __init__(self, mode="safe", approver=None):
        """mode is "read-only", "safe" or "yolo"; approver(call, reason) -> bool."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.approver = approver or _refuse

    def check(self, call):
        """Return None to allow the call, or a reason string to block it."""
        name = call["name"]
        if name == "bash":
            command = str(call["args"].get("command", ""))
            if any(re.search(p, command) for p in DENY_PATTERNS):
                return f"{command!r} is a destructive or privileged command and is never run"
        if name in READ_TOOLS or self.mode == "yolo":
            return None
        if self.mode == "read-only":
            return f"{name} is not allowed in read-only mode"
        if self.approver(call, f"{name} changes files or runs commands"):
            return None
        return f"{name} was not approved"

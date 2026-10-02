"""Tools: plain functions the model can call, fenced into one directory.

Concept: a tool is a name, a JSON schema the model reads, and a callable the
harness runs. The @tool decorator derives the schema from a function's own
signature, so writing a tool is writing a function. core_tools() is the
minimal kit a coding agent needs: read, write, edit, run, list, search.

Design rules:
  * Every parameter is a string: models fill strings reliably, and each tool
    converts what it needs, so bad input fails inside the tool as an error.
  * One gate for paths: resolve() refuses anything whose real path
    (symlinks followed) leaves the working directory.
  * Outputs are bounded, since every result lands in the context window.
  * Fixable mistakes (a bad edit snippet) return "ERROR: ..." advice;
    forbidden ones (escaping the fence) raise.
"""
import fnmatch
import inspect
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
MAX_READ_LINES = 4000
MAX_BASH_CHARS = 12000
MAX_LIST = 500
MAX_GREP_HITS = 200


@dataclass
class Tool:
    """A callable the model may invoke: name, provider spec, implementation."""
    name: str
    spec: dict
    run: Callable


def tool(description, **params):
    """Decorate a function into a Tool; params maps argument name -> description.

    Arguments with defaults become optional; all are declared as strings.
    """
    def wrap(fn):
        sig = inspect.signature(fn)
        properties = {name: {"type": "string", "description": params.get(name, name)}
                      for name in sig.parameters}
        required = [name for name, p in sig.parameters.items()
                    if p.default is inspect.Parameter.empty]
        parameters = {"type": "object", "properties": properties, "required": required}
        spec = {"schema": {"name": fn.__name__, "description": description,
                           "parameters": parameters}}
        return Tool(name=fn.__name__, spec=spec, run=fn)
    return wrap


def _matches(rel, pattern):
    """True when a relative path or its basename matches a glob pattern."""
    # fnmatch's "*" already crosses "/", so "**/" only adds a required
    # separator; dropping it lets "**/*.py" match top-level files too.
    loose = pattern[3:] if pattern.startswith("**/") else pattern
    return any(fnmatch.fnmatch(s, p) for s in (rel, os.path.basename(rel))
               for p in (pattern, loose))


def core_tools(workdir):
    """Return the six core tools, each confined to workdir."""
    root = os.path.realpath(workdir)

    def resolve(path):
        """Map a model-supplied path to a real path inside root, or raise."""
        full = os.path.realpath(os.path.join(root, path))
        # Compare whole path components (case-folded where the OS is), so
        # "/work-evil" does not pass as being inside "/work".
        base, target = os.path.normcase(root), os.path.normcase(full)
        if target != base and not target.startswith(base.rstrip(os.sep) + os.sep):
            raise PermissionError(f"{path!r} escapes the working directory")
        return full

    def walk():
        """Return sorted (relative_path, full_path) for every non-ignored file."""
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for full in (os.path.join(dirpath, name) for name in filenames):
                found.append((os.path.relpath(full, root).replace(os.sep, "/"), full))
        return sorted(found)

    @tool("Read a text file; returns lines numbered N<TAB>line.",
          path="File path relative to the working directory")
    def read_file(path):
        with open(resolve(path), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        out = [f"{i}\t{line}" for i, line in enumerate(lines[:MAX_READ_LINES], 1)]
        if len(lines) > MAX_READ_LINES:
            out.append(f"... truncated: file has {len(lines)} lines in total")
        return "\n".join(out)

    @tool("Create or overwrite a file with the given content.",
          path="File path relative to the working directory",
          content="Full new file content")
    def write_file(path, content):
        full = resolve(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    @tool("Replace one exact, unique snippet in a file.",
          path="File path relative to the working directory",
          old="Exact existing text, unique in the file", new="Replacement text")
    def edit_file(path, old, new):
        full = resolve(path)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        # Uniqueness rule: an ambiguous snippet could edit the wrong place,
        # so the model must quote enough context to pin down one location.
        count = text.count(old)
        if count == 0:
            return "ERROR: snippet not found — read the file and copy it exactly"
        if count > 1:
            return f"ERROR: snippet appears {count} times — include more context to make it unique"
        with open(full, "w", encoding="utf-8") as f:
            f.write(text.replace(old, new, 1))
        return f"Edited {path}"

    @tool("Run a shell command in the working directory; returns its output.",
          command="Shell command to run", timeout="Seconds before giving up (default 120)")
    def bash(command, timeout="120"):
        try:
            proc = subprocess.run(command, shell=True, cwd=root, capture_output=True,
                                  text=True, errors="replace", timeout=float(timeout))
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"
        out, half = (proc.stdout or "") + (proc.stderr or ""), MAX_BASH_CHARS // 2
        # Keep both ends: commands state intent first and report errors last.
        if len(out) > MAX_BASH_CHARS:
            out = f"{out[:half]}\n... [{len(out) - MAX_BASH_CHARS} chars truncated] ...\n{out[-half:]}"
        return out or f"(exit {proc.returncode}, no output)"

    @tool("List files matching a glob pattern, sorted.",
          pattern="Glob matched against relative path and basename (default **/*)")
    def list_files(pattern="**/*"):
        hits = [rel for rel, _ in walk() if _matches(rel, pattern)]
        if len(hits) > MAX_LIST:
            hits = hits[:MAX_LIST] + [f"... and {len(hits) - MAX_LIST} more"]
        return "\n".join(hits) or "(no files match)"

    @tool("Search file contents with a regular expression.",
          regex="Python regular expression", pattern="Glob limiting which files to search")
    def grep(regex, pattern="*"):
        rx = re.compile(regex)
        hits = []
        for rel, full in walk():
            if not _matches(rel, pattern):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    for n, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{n}: {line.rstrip()[:200]}")
                            if len(hits) >= MAX_GREP_HITS:
                                return "\n".join(hits + ["... hit cap of 200 reached"])
            except OSError:
                continue  # unreadable file: skip rather than abort the search
        return "\n".join(hits) or "(no matches)"

    return [read_file, write_file, edit_file, bash, list_files, grep]

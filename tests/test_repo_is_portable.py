"""The repository clones and runs somewhere that is not this machine.

Every tracked file used to be free to name one operator's filesystem, and
several did: the package's own default `state_dir`, both config files, two
benchmark scripts, and a sandbox-escape test that opened a database by
absolute path. On any other machine those defaults pointed nowhere -- and
worse, the sandbox test passed, because the file it tried to steal was not
there to refuse.

These are cheap to check and easy to reintroduce, so they are checked.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#   F:/hexylab/...        a working drive that exists on one machine
#   C:\Users\someone\...  a named home (the bare `C:\Users` root is not one)
#   /home/someone/...     the same on POSIX
#
# Deliberately these shapes and not "any absolute path". A broader rule was
# tried and matched `https://github.com`, `C:\Python311` and the literal
# `except Exception as e:\n` inside a test's source string -- a guard that
# cries wolf gets an allow-list bolted on until it means nothing. This one
# catches what actually went wrong and says so plainly.
MACHINE_PATH = re.compile(
    "[A-Za-z]:[\\\\/]{1,2}hexylab"
    "|[A-Za-z]:[\\\\/]{1,2}(?:Users|home)[\\\\/][A-Za-z0-9._-]+"
    "|/(?:home|Users)/[A-Za-z0-9._-]+",
    re.IGNORECASE,
)

# What a fresh clone runs from, plus the documentation, because setup
# instructions naming a drive nobody else has are instructions that cannot be
# followed. `USER_GUIDE.md` shows the shape the rest is held to:
# `C:/Users/YOUR_NAME/...` is a placeholder and passes; a real directory does
# not.
CHECKED_SUFFIXES = {".py", ".toml", ".cfg", ".ini", ".txt", ".json", ".yml",
                    ".yaml", ".md"}

# Placeholders, not locations. A guide has to show the shape of a path.
PLACEHOLDER = re.compile("YOUR_NAME|<install>|USERNAME|your-model",
                         re.IGNORECASE)


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [ROOT / name for name in out if name.strip()]


def test_the_check_itself_is_not_vacuous():
    """The pattern has to catch the thing that actually happened.

    Written first because it nearly did not: an earlier version of this file
    was created through a shell heredoc that ate the backslashes out of the
    character class, leaving a pattern that matched forward slashes only. It
    would have passed this whole module while seeing none of the Windows paths
    it exists for.
    """
    backslash = chr(92)
    assert MACHINE_PATH.search("state_dir = " + chr(34) + "F:/hexylab/x")
    assert MACHINE_PATH.search("db = r" + chr(34) + "F:" + backslash + "hexylab")
    assert MACHINE_PATH.search("C:" + backslash + "Users" + backslash + "pstry")
    assert MACHINE_PATH.search('"/home/pete/thing"')
    # And not the absolute paths that are nobody's machine in particular,
    # nor the several things a broader pattern mistook for one.
    assert not MACHINE_PATH.search('"C:/Windows/System32/drivers/etc/hosts"')
    assert not MACHINE_PATH.search("| List `C:" + backslash + "Users` | blocked |")
    assert not MACHINE_PATH.search("C:" + backslash + "Python311")
    assert not MACHINE_PATH.search("https://github.com/PStryder/amoeba")
    assert not MACHINE_PATH.search('"    except Exception as e:' + backslash + 'n"')
    assert not MACHINE_PATH.search("count=lambda t: len(t) // 3")


def test_no_tracked_file_names_a_machine_specific_path():
    offenders: list[str] = []
    for path in _tracked():
        # This file has to contain the shapes it looks for.
        if path.name == Path(__file__).name:
            continue
        if path.suffix.lower() not in CHECKED_SUFFIXES or not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not MACHINE_PATH.search(line):
                continue
            if PLACEHOLDER.search(line):
                continue
            where = path.relative_to(ROOT).as_posix()
            offenders.append(f"{where}:{number}: {line.strip()[:90]}")
    assert not offenders, (
        "tracked files name one machine's filesystem:\n  "
        + "\n  ".join(offenders))


def test_no_runtime_state_is_tracked():
    """No database, blob, log or sandbox file belongs in the history."""
    bad = [p.relative_to(ROOT).as_posix() for p in _tracked()
           if p.suffix.lower() in {".sqlite3", ".db", ".blob", ".log", ".jsonl"}
           or "/blobs/" in p.as_posix() or "/sandbox/" in p.as_posix()]
    assert not bad, f"runtime state is tracked: {bad}"


def test_the_operators_own_config_is_not_tracked():
    """`config.toml` names this machine; `config.example.toml` is the shipped one."""
    tracked = {p.relative_to(ROOT).as_posix() for p in _tracked()}
    assert "config.example.toml" in tracked, "there is nothing to copy from"
    for live in ("config.toml", "config.test.toml"):
        assert live not in tracked, f"{live} is an operator's own file"


def test_the_defaults_need_no_configuration_to_be_valid():
    """A checkout with no config at all still produces usable locations."""
    from amoeba.config import Config

    cfg = Config()
    for name in ("state_dir", "runtime_dir", "models_dir"):
        value = getattr(cfg, name)
        assert not value.is_absolute(), f"{name} defaults to an absolute path"
        assert MACHINE_PATH.search(str(value)) is None

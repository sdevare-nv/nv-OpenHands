#!/usr/bin/env python3
"""Standalone test script for command_blacklist.py

Run this script directly to test all blacklist patterns:
    python test_command_blacklist.py

Or run with verbose output:
    python test_command_blacklist.py -v
"""

import re
import sys
from dataclasses import dataclass


# ============================================================================
# Inline copy of the blacklist logic (for standalone testing without imports)
# ============================================================================


@dataclass
class BlacklistEntry:
    """A blacklisted command pattern with associated feedback."""

    pattern: str
    feedback: str
    description: str
    enabled: bool = True


COMMAND_BLACKLIST: list[BlacklistEntry] = [
    # === Block ALL killall commands ===
    BlacklistEntry(
        pattern=r"\bkillall\b",
        feedback="ERROR: The `killall` command is not allowed.",
        description="Blocks all killall commands",
    ),
    # === Block ALL pkill commands ===
    BlacklistEntry(
        pattern=r"\bpkill\b",
        feedback="ERROR: The `pkill` command is not allowed.",
        description="Blocks all pkill commands",
    ),
    # === Block kill with command substitution $(...) ===
    BlacklistEntry(
        pattern=r"\bkill\s+.*\$\(",
        feedback="ERROR: The `kill` command with command substitution is not allowed.",
        description="Blocks kill with command substitution $()",
    ),
    # === Block kill with backtick command substitution ===
    BlacklistEntry(
        pattern=r"\bkill\s+.*`",
        feedback="ERROR: The `kill` command with command substitution is not allowed.",
        description="Blocks kill with backtick command substitution",
    ),
    # === Block kill with variables ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*\$\w+",
        feedback="ERROR: The `kill` command with shell variables is not allowed.",
        description="Blocks kill with shell variables",
    ),
    # === Block kill -1 or kill -9 -1 ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*-1\b",
        feedback="ERROR: This command would kill all processes you own.",
        description="Blocks kill -1 which kills all user processes",
    ),
    # === Block kill 0 ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-\d+\s+|-[A-Z]+\s+|-SIG[A-Z]+\s+)*0\b",
        feedback="ERROR: This command would kill all processes in the current process group.",
        description="Blocks kill 0 which kills the process group",
    ),
    # === Block kill with negative PIDs ===
    BlacklistEntry(
        pattern=r"\bkill\s+(-[1-9]|-1[0-5]|-[A-Z]+|-SIG[A-Z]+)?\s*-([2-9]\d\d+|[1-9]\d\d\d+)\s*$",
        feedback="ERROR: The `kill` command with negative PIDs is not allowed.",
        description="Blocks kill with negative PIDs (process groups)",
    ),
    # === Dangerous rm commands ===
    BlacklistEntry(
        pattern=r"\brm\s+(-\w+\s+)*(/\s*$|/\*)",
        feedback="ERROR: This command would destroy the entire filesystem.",
        description="Blocks rm -rf / or rm -rf /*",
    ),
    BlacklistEntry(
        pattern=r"\brm\s+(-\w+\s+)*(/(bin|usr|etc|var|home|root|opt|lib|lib64|sbin|boot|dev|proc|sys))\s*$",
        feedback="ERROR: This command would delete critical system directories.",
        description="Blocks rm of critical system directories",
    ),
    # === tmux kill commands ===
    BlacklistEntry(
        pattern=r"\btmux\s+(kill-server|kill-session\s+-t\s+openhands)",
        feedback="ERROR: This command would terminate the OpenHands tmux session.",
        description="Blocks killing the openhands tmux session",
    ),
    # === Shutdown/reboot commands ===
    BlacklistEntry(
        pattern=r"\b(shutdown|reboot|poweroff|halt|init\s+[06])\b",
        feedback="ERROR: System shutdown/reboot commands are blocked.",
        description="Blocks system shutdown/reboot commands",
    ),
    # === Dangerous dd commands ===
    BlacklistEntry(
        pattern=r"\bdd\s+.*of=\s*(/dev/sd[a-z]|/dev/nvme\w*|/dev/hd[a-z]|/dev/null)\b",
        feedback="ERROR: This dd command could overwrite disk devices.",
        description="Blocks dd commands that write to disk devices",
    ),
]


def check_command(command: str) -> tuple[bool, str]:
    """Check if a command is blocked. Returns (is_blocked, description)."""
    for entry in COMMAND_BLACKLIST:
        if not entry.enabled:
            continue
        try:
            if re.search(entry.pattern, command.strip(), re.IGNORECASE):
                return True, entry.description
        except re.error:
            continue
    return False, ""


# ============================================================================
# Test Cases
# ============================================================================

# Format: (command, should_be_blocked, description)
TEST_CASES = [
    # === ALLOWED kill commands (specific PIDs) ===
    ("kill 12345", False, "kill with specific PID"),
    ("kill -9 12345", False, "kill -9 with specific PID"),
    ("kill -15 12345", False, "kill -15 with specific PID"),
    ("kill -TERM 12345", False, "kill -TERM with specific PID"),
    ("kill -SIGTERM 12345", False, "kill -SIGTERM with specific PID"),
    ("kill -9 12345 67890", False, "kill multiple specific PIDs"),
    ("kill %1", False, "kill job spec (allowed)"),
    ("kill %2", False, "kill job spec %2 (allowed)"),
    # === BLOCKED killall commands ===
    ("killall python", True, "killall python"),
    ("killall -9 python", True, "killall -9 python"),
    ("killall uvicorn", True, "killall uvicorn"),
    ("killall node", True, "killall node"),
    ("killall -TERM myprocess", True, "killall with signal"),
    # === BLOCKED pkill commands ===
    ("pkill python", True, "pkill python"),
    ("pkill -9 python", True, "pkill -9 python"),
    ("pkill -f python", True, "pkill -f python"),
    ("pkill -f 'my script'", True, "pkill -f with pattern"),
    ("pkill uvicorn", True, "pkill uvicorn"),
    ("pkill -f server", True, "pkill -f server"),
    # === BLOCKED kill with command substitution ===
    ("kill $(pgrep python)", True, "kill with $() substitution"),
    ("kill -9 $(pgrep python)", True, "kill -9 with $() substitution"),
    ("kill $(cat /tmp/pid)", True, "kill with $() reading file"),
    ("kill `pgrep python`", True, "kill with backtick substitution"),
    ("kill -9 `pgrep python`", True, "kill -9 with backtick substitution"),
    # === BLOCKED kill with variables ===
    ("kill $PID", True, "kill with variable"),
    ("kill -9 $PID", True, "kill -9 with variable"),
    ("kill $my_pid", True, "kill with underscore variable"),
    ("kill -TERM $PID", True, "kill -TERM with variable"),
    # === BLOCKED kill -1 (all processes) ===
    ("kill -1", True, "kill -1"),
    ("kill -9 -1", True, "kill -9 -1"),
    ("kill -TERM -1", True, "kill -TERM -1"),
    ("kill -SIGKILL -1", True, "kill -SIGKILL -1"),
    # === BLOCKED kill 0 (process group) ===
    ("kill 0", True, "kill 0"),
    ("kill -9 0", True, "kill -9 0"),
    ("kill -TERM 0", True, "kill -TERM 0"),
    # === BLOCKED kill with negative PIDs (as target, not signal) ===
    ("kill -12345", True, "kill -12345 (negative PID as target)"),
    ("kill -9 -12345", True, "kill -9 -12345 (negative PID as target)"),
    ("kill -200", True, "kill -200 (negative PID as target)"),
    # === BLOCKED rm commands ===
    ("rm -rf /", True, "rm -rf /"),
    ("rm -rf /*", True, "rm -rf /*"),
    ("rm -r /", True, "rm -r /"),
    ("rm /bin", True, "rm /bin"),
    ("rm -rf /usr", True, "rm -rf /usr"),
    ("rm -rf /etc", True, "rm -rf /etc"),
    ("rm -rf /var", True, "rm -rf /var"),
    ("rm /home", True, "rm /home"),
    ("rm -rf /root", True, "rm -rf /root"),
    # === ALLOWED rm commands ===
    ("rm -rf /tmp/mydir", False, "rm -rf /tmp/mydir (allowed)"),
    ("rm file.txt", False, "rm file.txt (allowed)"),
    ("rm -rf ./build", False, "rm -rf ./build (allowed)"),
    ("rm -rf /workspace/project", False, "rm -rf /workspace/project (allowed)"),
    # === BLOCKED tmux commands ===
    ("tmux kill-server", True, "tmux kill-server"),
    ("tmux kill-session -t openhands", True, "tmux kill-session -t openhands"),
    # === ALLOWED tmux commands ===
    (
        "tmux kill-session -t mysession",
        False,
        "tmux kill-session -t mysession (allowed)",
    ),
    ("tmux new-session", False, "tmux new-session (allowed)"),
    ("tmux list-sessions", False, "tmux list-sessions (allowed)"),
    # === BLOCKED shutdown/reboot commands ===
    ("shutdown", True, "shutdown"),
    ("shutdown -h now", True, "shutdown -h now"),
    ("reboot", True, "reboot"),
    ("poweroff", True, "poweroff"),
    ("halt", True, "halt"),
    ("init 0", True, "init 0"),
    ("init 6", True, "init 6"),
    # === BLOCKED dd commands ===
    ("dd if=/dev/zero of=/dev/sda", True, "dd to /dev/sda"),
    ("dd if=/dev/zero of=/dev/nvme0n1", True, "dd to /dev/nvme"),
    ("dd if=/dev/zero of=/dev/null", True, "dd to /dev/null"),
    # === ALLOWED dd commands ===
    (
        "dd if=/dev/zero of=./testfile bs=1M count=10",
        False,
        "dd to regular file (allowed)",
    ),
    # === Other allowed commands ===
    ("ls -la", False, "ls -la (allowed)"),
    ("ps aux", False, "ps aux (allowed)"),
    ("ps aux | grep python", False, "ps aux | grep python (allowed)"),
    ("cat /etc/passwd", False, "cat /etc/passwd (allowed)"),
    ("echo hello", False, "echo hello (allowed)"),
    ("python script.py", False, "python script.py (allowed)"),
]


# ============================================================================
# Test Runner
# ============================================================================


def run_tests(verbose: bool = False) -> tuple[int, int, list]:
    """Run all test cases and return (passed, failed, failures)."""
    passed = 0
    failed = 0
    failures = []

    for command, should_block, description in TEST_CASES:
        is_blocked, matched_desc = check_command(command)

        if is_blocked == should_block:
            passed += 1
            if verbose:
                status = "BLOCKED" if is_blocked else "ALLOWED"
                print(f"  ✓ {status}: {command!r} - {description}")
        else:
            failed += 1
            expected = "BLOCKED" if should_block else "ALLOWED"
            actual = "BLOCKED" if is_blocked else "ALLOWED"
            failures.append(
                {
                    "command": command,
                    "description": description,
                    "expected": expected,
                    "actual": actual,
                    "matched_rule": matched_desc if is_blocked else None,
                }
            )
            if verbose:
                print(f"  ✗ FAIL: {command!r}")
                print(f"      Expected: {expected}, Got: {actual}")
                if is_blocked:
                    print(f"      Matched: {matched_desc}")

    return passed, failed, failures


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print("=" * 70)
    print("Command Blacklist Test Suite")
    print("=" * 70)
    print()

    # Show all blacklist rules
    print("Blacklist Rules:")
    print("-" * 70)
    for i, entry in enumerate(COMMAND_BLACKLIST, 1):
        print(f"  {i}. {entry.description}")
        if verbose:
            print(f"     Pattern: {entry.pattern}")
    print()

    # Run tests
    print("Running Tests:")
    print("-" * 70)
    passed, failed, failures = run_tests(verbose)
    print()

    # Summary
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failures:
        print()
        print("FAILURES:")
        print("-" * 70)
        for f in failures:
            print(f"  Command: {f['command']!r}")
            print(f"  Description: {f['description']}")
            print(f"  Expected: {f['expected']}, Got: {f['actual']}")
            if f["matched_rule"]:
                print(f"  Matched Rule: {f['matched_rule']}")
            print()

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

"""Recursive file-search stress test environment for Memex(RL).

Creates a simulated filesystem with 1000 mock documents across
nested directories. The agent must search through files to find
a target piece of information, testing the Memex memory system's
ability to handle long-horizon search under context pressure.

This is the stress test described in Section 4.3 of the paper.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from src.environments.base import Environment, StepResult


@dataclass
class MockFile:
    """A mock document in the simulated filesystem."""
    path: str
    content: str
    size: int  # simulated size in bytes


class StressTestEnv(Environment):
    """Recursive file-search stress test with 1000 documents.

    The agent is given a target query and must search through a
    nested directory structure to find the matching document.
    Each directory listing and file read adds to the context,
    forcing the agent to compress aggressively.

    Args:
        num_files: Total number of files to generate.
        num_dirs: Number of top-level directories.
        max_depth: Maximum directory nesting depth.
        target_depth: Depth at which the target file is placed.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_files: int = 1000,
        num_dirs: int = 10,
        max_depth: int = 5,
        target_depth: int = 3,
        seed: int = 42,
    ) -> None:
        self._num_files = num_files
        self._num_dirs = num_dirs
        self._max_depth = max_depth
        self._target_depth = target_depth
        self._seed = seed

        self._files: dict[str, MockFile] = {}
        self._dirs: dict[str, list[str]] = {}  # dir → children
        self._target_path: str = ""
        self._target_content: str = ""
        self._target_keyword: str = ""
        self._task_id: str = ""
        self._done: bool = False
        self._found: bool = False

    def reset(self, task_id: str | None = None) -> str:
        """Generate filesystem and return the search task."""
        self._done = False
        self._found = False
        self._task_id = task_id or f"stress_test_{self._seed}"

        rng = random.Random(self._seed)
        self._generate_filesystem(rng)

        return (
            f"Find the document containing the keyword '{self._target_keyword}' "
            f"in the filesystem. The filesystem has {self._num_files} files across "
            f"{self._num_dirs} top-level directories with nesting up to depth "
            f"{self._max_depth}. Use 'ls <path>' to list directories and "
            f"'cat <path>' to read files. Reply with finish when found."
        )

    def step(self, action: str) -> StepResult:
        """Execute a filesystem action."""
        if self._done:
            return StepResult(observation="Episode finished.", done=True,
                              reward=1.0 if self._found else 0.0)

        action = action.strip()

        if action.startswith("ls "):
            return self._handle_ls(action[3:].strip())
        elif action.startswith("cat "):
            return self._handle_cat(action[4:].strip())
        elif action.startswith("grep ") or action.startswith("search "):
            return self._handle_grep(action)
        elif action.startswith("find "):
            return self._handle_find(action[5:].strip())
        else:
            return StepResult(
                observation=(
                    "Available commands: ls <dir>, cat <file>, "
                    "grep <keyword> <dir>, find <pattern>"
                ),
            )

    def get_task_id(self) -> str:
        return self._task_id

    @property
    def is_done(self) -> bool:
        return self._done

    # ── Filesystem generation ──────────────────────────────────────────

    def _generate_filesystem(self, rng: random.Random) -> None:
        """Generate a mock filesystem with nested directories and files."""
        self._files = {}
        self._dirs = {"/": []}

        # Topics for generating realistic-looking documents
        topics = [
            "machine learning", "database optimization", "network security",
            "cloud architecture", "data pipelines", "API design",
            "distributed systems", "performance tuning", "code review",
            "testing strategies", "deployment automation", "monitoring",
        ]
        formats = ["report", "notes", "analysis", "summary", "review", "spec"]

        # Generate directory structure
        all_dirs = ["/"]
        for i in range(self._num_dirs):
            dirname = f"/dept_{i:02d}"
            self._dirs["/"].append(dirname)
            self._dirs[dirname] = []
            all_dirs.append(dirname)

            for depth in range(1, self._max_depth):
                parent = dirname + "/" + "/".join(f"sub_{d}" for d in range(depth))
                child = parent + f"/sub_{depth}"
                if parent not in self._dirs:
                    self._dirs[parent] = []
                self._dirs[parent].append(child)
                self._dirs[child] = []
                all_dirs.append(child)

        # Place target file at specified depth
        target_dir = f"/dept_{rng.randint(0, self._num_dirs - 1):02d}"
        for d in range(1, self._target_depth + 1):
            target_dir += f"/sub_{d}"
        if target_dir not in self._dirs:
            self._dirs[target_dir] = []

        # Generate the target keyword and file
        self._target_keyword = f"ALPHA_{rng.randint(10000, 99999)}"
        target_filename = f"classified_{rng.choice(formats)}.txt"
        self._target_path = f"{target_dir}/{target_filename}"
        self._target_content = (
            f"Project {self._target_keyword} Status Report\n"
            f"Classification: CONFIDENTIAL\n"
            f"This document contains the target information for keyword "
            f"'{self._target_keyword}'. The project milestone was achieved.\n"
            f"Budget allocation: $2.4M approved for Q3.\n"
        )
        self._files[self._target_path] = MockFile(
            path=self._target_path,
            content=self._target_content,
            size=len(self._target_content),
        )
        if target_dir in self._dirs:
            self._dirs[target_dir].append(self._target_path)

        # Generate remaining files
        files_placed = 1
        while files_placed < self._num_files:
            dir_path = rng.choice(all_dirs)
            topic = rng.choice(topics)
            fmt = rng.choice(formats)
            filename = f"{topic.replace(' ', '_')}_{fmt}_{files_placed:04d}.txt"
            filepath = f"{dir_path}/{filename}"

            # Generate filler content (doesn't contain target keyword)
            content = (
                f"{topic.title()} {fmt.title()}\n"
                f"Document ID: DOC-{files_placed:04d}\n"
                f"This document covers {topic} considerations.\n"
                f"{'Lorem ipsum ' * rng.randint(20, 80)}\n"
            )
            self._files[filepath] = MockFile(
                path=filepath, content=content, size=len(content)
            )
            if dir_path in self._dirs:
                self._dirs[dir_path].append(filepath)
            files_placed += 1

    # ── Command handlers ───────────────────────────────────────────────

    def _handle_ls(self, path: str) -> StepResult:
        path = path.rstrip("/") or "/"
        if path not in self._dirs:
            return StepResult(observation=f"Directory not found: {path}")

        children = self._dirs[path]
        if not children:
            return StepResult(observation=f"{path}/ (empty directory)")

        # Separate dirs and files
        dirs = sorted(c for c in children if c in self._dirs)
        files = sorted(c for c in children if c in self._files)

        lines = [f"Contents of {path}/ ({len(dirs)} dirs, {len(files)} files):"]
        for d in dirs[:20]:  # Limit output
            name = d.split("/")[-1]
            lines.append(f"  📁 {name}/")
        for f in files[:30]:
            name = f.split("/")[-1]
            size = self._files[f].size
            lines.append(f"  📄 {name} ({size}B)")
        if len(dirs) > 20 or len(files) > 30:
            lines.append(f"  ... and {max(0, len(dirs)-20) + max(0, len(files)-30)} more")

        return StepResult(observation="\n".join(lines))

    def _handle_cat(self, path: str) -> StepResult:
        if path not in self._files:
            return StepResult(observation=f"File not found: {path}")

        content = self._files[path].content
        # Check if agent found the target
        if self._target_keyword in content:
            self._found = True

        return StepResult(observation=content)

    def _handle_grep(self, action: str) -> StepResult:
        """Simple grep across files in a directory."""
        parts = action.split(maxsplit=2)
        if len(parts) < 3:
            return StepResult(
                observation="Usage: grep <keyword> <directory>"
            )
        keyword = parts[1]
        search_dir = parts[2].rstrip("/")

        matches = []
        for path, f in self._files.items():
            if path.startswith(search_dir) and keyword.lower() in f.content.lower():
                matches.append(path)

        if not matches:
            return StepResult(
                observation=f"No files matching '{keyword}' in {search_dir}/"
            )
        return StepResult(
            observation=(
                f"Found {len(matches)} file(s) matching '{keyword}' in {search_dir}/:\n"
                + "\n".join(f"  {m}" for m in matches[:10])
                + ("\n  ... and more" if len(matches) > 10 else "")
            )
        )

    def _handle_find(self, pattern: str) -> StepResult:
        """Find files matching a name pattern."""
        matches = [
            p for p in self._files
            if pattern.lower() in p.lower()
        ]
        if not matches:
            return StepResult(observation=f"No files matching '{pattern}'")
        return StepResult(
            observation=(
                f"Found {len(matches)} file(s) matching '{pattern}':\n"
                + "\n".join(f"  {m}" for m in sorted(matches)[:15])
                + ("\n  ... and more" if len(matches) > 15 else "")
            )
        )

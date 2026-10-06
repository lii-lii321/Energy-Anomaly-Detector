import re
from pathlib import Path

from conftest import REPO_ROOT

README_PATH = Path(REPO_ROOT) / "README.md"
PATH_EXTENSIONS = (".py", ".pkl", ".png", ".txt", ".md", ".json", ".csv")
IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
TREE_ENTRY = re.compile(r"^(?P<indent>(?:│   |    )*)(?:├── |└── )(?P<name>.+)$")


def _readme_text():
    assert README_PATH.exists(), f"missing README at {README_PATH}"
    return README_PATH.read_text(encoding="utf-8")


def _fenced_block_after(text, heading_marker):
    lines = text.splitlines()
    heading = next(
        (
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith("#") and heading_marker in line
        ),
        None,
    )
    if heading is None:
        return None
    opening = next(
        (i for i in range(heading + 1, len(lines)) if lines[i].strip().startswith("```")),
        None,
    )
    if opening is None:
        return None
    closing = next(
        (i for i in range(opening + 1, len(lines)) if lines[i].strip().startswith("```")),
        None,
    )
    if closing is None:
        return None
    return lines[opening + 1 : closing]


def _bash_fenced_blocks(text):
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("```bash"):
            closing = next(
                (
                    i
                    for i in range(index + 1, len(lines))
                    if lines[i].strip().startswith("```")
                ),
                None,
            )
            assert closing is not None, "unterminated bash code fence in README"
            blocks.append(lines[index + 1 : closing])
            index = closing + 1
        else:
            index += 1
    return blocks


def _structure_tree_entries(block_lines):
    entries = []
    prefixes = {-1: ""}
    for line in block_lines:
        if not line.strip():
            continue
        match = TREE_ENTRY.match(line)
        if match is None:
            assert not entries, f"unparsable structure tree line: {line!r}"
            continue
        depth = len(match.group("indent")) // 4
        name = match.group("name").split("#")[0].strip()
        parent = prefixes.get(depth - 1, "")
        rel = f"{parent}{name}" if parent else name
        if name.endswith("/"):
            prefixes[depth] = rel
        entries.append((rel, name.endswith("/")))
    return entries


def test_readme_exists():
    assert README_PATH.exists()


def test_stale_project_name_is_gone():
    assert "Energy-Anomaly-Detection/" not in _readme_text()


DATA_PATH_PATTERN = re.compile(r"\bdata/[\w./-]*")


def test_data_dir_references_exist():
    for match in DATA_PATH_PATTERN.finditer(_readme_text()):
        referenced = match.group(0).rstrip(".")
        assert (Path(REPO_ROOT) / referenced).exists(), (
            f"README references missing data path: {referenced}"
        )


def test_structure_tree_entries_exist():
    block = _fenced_block_after(_readme_text(), "项目结构")
    assert block is not None, "README must contain the project structure code block"
    entries = _structure_tree_entries(block)
    assert entries, "project structure tree must list entries"
    for rel, is_dir in entries:
        target = Path(REPO_ROOT) / rel
        if is_dir:
            assert target.is_dir(), f"structure tree directory does not exist: {rel}"
        else:
            assert target.is_file(), f"structure tree file does not exist: {rel}"


def test_image_links_exist():
    for target in IMAGE_LINK.findall(_readme_text()):
        if target.startswith(("http://", "https://")):
            continue
        assert (Path(REPO_ROOT) / target).is_file(), f"README references missing image: {target}"


def test_inline_path_references_exist():
    for token in INLINE_CODE.findall(_readme_text()):
        looks_like_path = (
            "/" in token
            and not token.startswith("/")
            and " " not in token
            and ":" not in token
            and token.endswith(PATH_EXTENSIONS)
        )
        if looks_like_path:
            assert (Path(REPO_ROOT) / token).is_file(), f"README references missing path: {token}"


def test_quick_start_scripts_exist():
    for block in _bash_fenced_blocks(_readme_text()):
        for line in block:
            for token in line.split():
                if token.endswith(".py"):
                    assert (Path(REPO_ROOT) / token).is_file(), (
                        f"quick start references missing script: {token}"
                    )

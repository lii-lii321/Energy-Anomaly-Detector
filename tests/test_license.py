from pathlib import Path

from conftest import REPO_ROOT

LICENSE_PATH = Path(REPO_ROOT) / "LICENSE"
README_PATH = Path(REPO_ROOT) / "README.md"


def test_license_file_is_standard_mit():
    assert LICENSE_PATH.is_file(), f"missing LICENSE at {LICENSE_PATH}"
    content = LICENSE_PATH.read_text(encoding="utf-8")
    assert "MIT License" in content, "LICENSE must declare the MIT License"
    assert "Copyright (c) 2026" in content, "LICENSE must carry the copyright line"


def test_readme_states_mit_license_and_links_file():
    content = README_PATH.read_text(encoding="utf-8")
    assert "MIT" in content, "README must state the project is MIT licensed"
    assert "[LICENSE](./LICENSE)" in content, "README must link to ./LICENSE"

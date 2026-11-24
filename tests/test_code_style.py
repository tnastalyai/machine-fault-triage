import io
import tokenize
from pathlib import Path

import pytest

SOURCE_ROOTS = ("src", "tests")
ALLOWED_PREFIXES = ("# type:", "# noqa", "# pragma:", "# workaround:")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def python_files() -> list[Path]:
    return sorted(path for root in SOURCE_ROOTS for path in (PROJECT_ROOT / root).rglob("*.py"))


def comments_in(path: Path) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
    return [token.string for token in tokens if token.type == tokenize.COMMENT]


@pytest.mark.parametrize("path", python_files(), ids=lambda path: path.name)
def test_code_explains_itself_without_comments(path: Path) -> None:
    unexpected = [
        comment for comment in comments_in(path) if not comment.startswith(ALLOWED_PREFIXES)
    ]

    assert unexpected == []

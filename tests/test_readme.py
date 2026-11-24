import contextlib
import io
import re
from pathlib import Path

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def fenced_blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", README, flags=re.DOTALL)


def test_quickstart_runs_and_prints_the_documented_output() -> None:
    (quickstart,) = fenced_blocks("python")
    (documented_output,) = fenced_blocks("text")
    printed = io.StringIO()

    with contextlib.redirect_stdout(printed):
        exec(compile(quickstart, "README.md", "exec"), {"__name__": "__main__"})  # noqa: S102

    assert printed.getvalue() == documented_output

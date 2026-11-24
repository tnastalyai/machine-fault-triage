# Contributing

## Setup

```bash
uv sync
uv run pre-commit install
```

## Checks

All of these run in CI and must pass:

```bash
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest --cov
```

## Code conventions

- Code carries no comments. Names, small functions and types explain intent. The only exception is a `# workaround:` comment that links to the upstream issue it works around. A test enforces this.
- Public modules, classes and functions have Google-style docstrings. Private helpers do not.
- Everything is fully typed and checked with `mypy --strict`.
- Design rationale belongs in `docs/adr/`, not in the code.
- Every behaviour change comes with a test.

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): description`, imperative mood.
- Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `build`, `chore`.
- Scopes follow the package layout: `models`, `ingest`, `retrieval`, `diagnosis`, `grounding`, `evals`, `safety`, `cli`.
- One logical change per commit, tests included.
- Pull requests are squash-merged; the PR title follows the same convention.

## Evaluation changes

A change to prompts, models or retrieval is accepted on the basis of evaluation results, reported with their uncertainty. Include the before and after numbers and the command that reproduces them.

## Use of AI tools

AI coding assistants are used in the development of this project. Every change is reviewed, tested and owned by its human author, and commits carry no tool attribution.

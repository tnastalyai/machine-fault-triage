# machine-fault-triage

Evidence-grounded fault triage for field service: heat pumps, HVAC, agricultural machinery.

Given a fault report, the device's manuals and its service history, the library returns a structured hint for the technician: probable causes, checks to run, the likely fix and parts to bring. Every cause points to a verified quote from a manual page or to a past service record. When the evidence is missing, the result says so instead of guessing.

> **Status: pre-alpha.** A baseline run (all evidence in one model call, then citation verification), an evaluation harness and a Claude adapter exist. The adapter is tested against a stubbed API only, so no evaluation results are published yet. Retrieval and document ingestion do not exist.

## Why

A technician usually learns what is wrong only after arriving on site, although the error code, the manual page that explains it and last spring's identical repair already exist. Generating a plausible answer from them is easy. Knowing when the answer is supported is the hard part, and the design is built around it:

- **Grounded or flagged.** The model answers in a draft schema that has no `verified` field. The library checks every quote against the cited page and every service record against the supplied history. A cause without verified evidence drops to low confidence ([ADR 0001](docs/adr/0001-verify-citations-outside-the-model.md)).
- **Abstention is a result.** `not_enough_data` with questions for the customer is a first-class outcome.
- **Untrusted input stays data.** Fault reports and document text reach the model only as `UntrustedBlock` values, apart from the instructions.
- **Bring your own model and storage.** `StructuredModel`, `DocumentProvider` and `HistoryProvider` are `typing.Protocol` interfaces. The core depends on no provider SDK.

## Quickstart

Runs offline: `FakeModel` returns a scripted draft, and the library verifies it.

```python
from machine_fault_triage import Document, FaultReport, Page, ServiceRecord, run_triage
from machine_fault_triage.drafts import (
    DraftCause,
    DraftDocumentCitation,
    DraftHistoryCitation,
    DraftTriage,
)
from machine_fault_triage.models import Confidence
from machine_fault_triage.testing import FakeModel

manual = Document(
    id="service-manual",
    title="AquaHeat 12 service manual",
    pages=(Page(number=77, text="7.3 Error codes\nE7: flow sensor open or short circuit"),),
)


class Evidence:
    def documents_for(self, *, tenant_id: str, device_id: str) -> list[Document]:
        return [manual]

    def records_for(self, *, tenant_id: str, device_id: str) -> list[ServiceRecord]:
        return []


model = FakeModel(
    [
        DraftTriage(
            summary="E7 points to the flow sensor.",
            enough_evidence=True,
            causes=(
                DraftCause(
                    title="Flow sensor fault",
                    confidence=Confidence.HIGH,
                    why="The manual maps E7 to the flow sensor.",
                    citations=(
                        DraftDocumentCitation(
                            document_id="service-manual",
                            page=77,
                            quote="E7: flow sensor open or short circuit",
                        ),
                        DraftHistoryCitation(record_id="order-404"),
                    ),
                ),
            ),
        )
    ]
)

result = run_triage(
    FaultReport(description="Display shows E7, no heating.", language="en"),
    tenant_id="acme",
    device_id="heat-pump-17",
    model=model,
    documents=Evidence(),
    history=Evidence(),
)

citation = result.causes[0].citations[0]
print(result.evidence_level, citation.document_title, citation.page, citation.verified)
```

```text
manual AquaHeat 12 service manual 77 True
```

The quote is on page 77, so the citation is verified. The cited service record `order-404` was never supplied, so that citation is dropped.

## Evaluation

Changes to prompts, models or retrieval are judged on a dataset, with metrics computed in code: answering or abstaining as the evidence allows, the expected top cause, the expected page or record, the share of citations that verify, resistance to prompt injection, and safety warnings ([ADR 0002](docs/adr/0002-score-evaluations-in-code.md)).

```bash
uv sync --all-extras
uv run machine-fault-triage eval run --dataset datasets/seed \
    --model machine_fault_triage.adapters.claude:ClaudeModel \
    --option model=claude-sonnet-5-5 --repeats 3 --out runs/sonnet.json
uv run machine-fault-triage eval compare runs/baseline.json runs/sonnet.json
```

`--model` takes any `module:callable` that returns a `StructuredModel`. Adapters for Claude and for the OpenAI Responses API ship as the `anthropic` and `openai` extras; both read credentials from the environment and call paid APIs. [`datasets/seed`](datasets/seed) holds two fictional devices and fifteen cases, three of them prompt-injection attempts. It is a first signal, too small to rank models that are close to each other.

## Limitations

- Verification proves that a quote exists on the page, not that it supports the cause.
- Quotes must be literal; text extracted with OCR errors fails to match.
- Top-cause accuracy is a keyword match, a proxy that a judging model will later complement.

## Roadmap

1. First measured baseline: accuracy, cost and latency
2. Behaviour without documents
3. Page-aware ingestion and hybrid retrieval
4. Tool-using agent variant, compared against the fixed pipeline
5. Postgres adapter and full evaluation report

## Development

```bash
uv sync --all-extras
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest --cov
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions. Licensed under [Apache-2.0](LICENSE).

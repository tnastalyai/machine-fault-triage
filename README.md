# machine-fault-triage

Evidence-grounded fault triage for field service: heat pumps, HVAC, agricultural machinery.

Given a fault report (free text, optional photos of a display or nameplate), the device's manuals and its service history, the library returns a structured hint for the technician: probable causes, checks to run, the likely fix and parts to bring. Every cause points to a verified quote from a manual page or to a past service record. When the evidence is missing, the result says so instead of guessing.

> **Status: pre-alpha.** A baseline run works end to end: all evidence in one model call, followed by citation verification. Retrieval, model adapters and the evaluation harness are being built in the order listed under [Roadmap](#roadmap). Nothing here is ready for production use.

## Why

A technician usually learns what is wrong only after arriving on site. The information needed for a first hypothesis often already exists: the error code is on a photo, its meaning is on page 77 of a manual, and the same model failed the same way last spring. This library connects those three sources before the visit.

The hard part is not generating a plausible answer. It is knowing when the answer is supported, and saying "not enough data" when it is not. The design is built around that constraint.

## Design principles

- **Grounded or flagged.** A cause is either backed by a verified citation or reported as general knowledge with low confidence.
- **Abstention is a valid result.** `not_enough_data` with questions for the customer is a first-class outcome, not an error.
- **Untrusted input stays data.** Fault reports and document text never become instructions. Tools are read-only.
- **Measured, not assumed.** Every change to prompts, models or retrieval is judged against an evaluation set with reported uncertainty.
- **Bring your own storage and model.** Documents, vector store, service history and the model provider are interfaces.

## Quickstart

The example runs offline: `FakeModel` stands in for a real model and returns a scripted draft. The library then checks every citation against the evidence it was given.

```python
from datetime import date

from machine_fault_triage import (
    Document,
    FaultReport,
    HistoryScope,
    Page,
    ServiceRecord,
    run_triage,
)
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
last_repair = ServiceRecord(
    id="order-9",
    recorded_on=date(2026, 3, 11),
    scope=HistoryScope.DEVICE,
    summary="E7 on display, flow sensor replaced",
)


class Evidence:
    def documents_for(self, *, tenant_id: str, device_id: str) -> list[Document]:
        return [manual]

    def records_for(self, *, tenant_id: str, device_id: str) -> list[ServiceRecord]:
        return [last_repair]


model = FakeModel(
    [
        DraftTriage(
            summary="E7 points to the flow sensor.",
            enough_evidence=True,
            causes=(
                DraftCause(
                    title="Flow sensor fault",
                    confidence=Confidence.HIGH,
                    why="The manual maps E7 to the flow sensor and it failed before.",
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

The quote was found on page 77, so the citation is verified. The model also cited service record `order-404`, which was never supplied; that citation is dropped.

## How grounding works

- The model answers in a draft schema that has no `verified` field. It cannot vouch for itself.
- A document citation is verified only if its quote occurs on the cited page of a supplied document.
- A history citation is verified only if the record was part of the supplied evidence.
- A cause left without a verified citation is reduced to low confidence and reported as general knowledge.
- Each verified citation names the document title, the page and the exact quote, so a technician can check it.

The reasoning is recorded in [ADR 0001](docs/adr/0001-verify-citations-outside-the-model.md).

## Bring your own model and data

The model, the document source and the service history are `typing.Protocol` interfaces: `StructuredModel`, `DocumentProvider` and `HistoryProvider`. Any object with the right methods works, and no provider SDK is a dependency. Fault reports and document text reach the model only as `UntrustedBlock` values, separate from the instructions.

## Roadmap

1. Result models, ports and a baseline run with citation verification
2. Evaluation harness and a hand-written seed dataset
3. Baseline: whole documents in the prompt
4. Behaviour without documents: abstention and questions for the customer
5. Page-aware ingestion and hybrid retrieval
6. Quote verification and prompt-injection hardening
7. Device and model service history
8. Tool-using agent variant, compared against the fixed pipeline
9. Postgres adapter and full evaluation report

## Development

```bash
uv sync
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest --cov
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions.

## License

[Apache-2.0](LICENSE)

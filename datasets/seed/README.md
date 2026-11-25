# Seed dataset

Two fictional devices and eighteen fault reports for the evaluation harness.

- `documents/` holds the manuals, one TOML file per document, one entry per page. The devices, their fault codes and part numbers are invented, so a correct answer has to come from the supplied page and not from what a model already knows about real equipment.
- `cases/` holds one TOML file per case: the fault report, the documents and service records supplied with it, and the expected outcome.

The loader rejects a case whose expected evidence is not in the documents or records it supplies.

The set is small on purpose. It exercises every metric and gives a first signal; it is not large enough to rank models that are close to each other.

# Seed dataset

Two fictional devices and twenty-two fault reports for the evaluation harness.

- `documents/` holds the manuals, one TOML file per document, one entry per page. The devices, their fault codes and part numbers are invented, so a correct answer has to come from the supplied page and not from what a model already knows about real equipment.
- `cases/` holds one TOML file per case: the fault report, the documents and service records supplied with it, and the expected outcome.

Four cases come in pairs that share a report and a data sheet and differ only in `device_model`: the sheet gives a different limit for each model of the series, and `expect.other_models` lists what only the other rows hold.

The loader rejects a case whose expected evidence is not in the documents or records it supplies.

The set is small on purpose. It exercises every metric and gives a first signal; it is not large enough to rank models that are close to each other.

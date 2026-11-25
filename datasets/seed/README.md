# Seed dataset

Two fictional devices and twenty-six fault reports for the evaluation harness.

- `documents/` holds the manuals, one TOML file per document, one entry per page. The devices, their fault codes and part numbers are invented, so a correct answer has to come from the supplied page and not from what a model already knows about real equipment.
- `cases/` holds one TOML file per case: the fault report, the documents and service records supplied with it, and the expected outcome.

Eight cases come in pairs that share a report and a data sheet and differ only in `device_model`. In four the sheet has a sentence for each model of the series; in the other four it has a table with the models in the header and the values in columns, the way a PDF table reads once it is text. `expect.other_models` lists what the answer would say only if it read another model's values.

The loader rejects a case whose expected evidence is not in the documents or records it supplies.

The set is small on purpose. It exercises every metric and gives a first signal; it is not large enough to rank models that are close to each other.

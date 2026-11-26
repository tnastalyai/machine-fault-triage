# Seed dataset

Two fictional devices and thirty-three fault reports for the evaluation harness.

- `documents/` holds the manuals, one TOML file per document, one entry per page. The devices, their fault codes and part numbers are invented, so a correct answer has to come from the supplied page and not from what a model already knows about real equipment.
- `cases/` holds one TOML file per case: the fault report, the documents and service records supplied with it, and the expected outcome.
- `photos/` holds the pictures that three reports attach: drawings of a display with one code each, made with Pillow's built-in font. They are not photographs, so they say whether a reading reaches the answer, not how well a reader copes with glare or a tilted phone.

Eight cases come in pairs that share a report and a data sheet and differ only in `device_model`. In four the sheet has a sentence for each model of the series; in the other four it has a table with the models in the header and the values in columns, the way a PDF table reads once it is text. `expect.other_models` lists what the answer would say only if it read another model's values.

One case needs two pages that follow each other: the code and the first check are on one page of the sprayer manual, the check that names the cause is on the next, which does not repeat the code, and the customer calls the boom "arms". Ranked by wording, the second page comes sixth of eight; with one neighbouring page on each side it is sent within a limit of four.

The loader rejects a case whose expected evidence is not in the documents or records it supplies.

The set is small on purpose. It exercises every metric and gives a first signal; it is not large enough to rank models that are close to each other.

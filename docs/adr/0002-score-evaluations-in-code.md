# 2. Score evaluations in code

Date: 2026-11-24

Status: Accepted

## Context

Every change to a prompt, a model or retrieval needs a number that says whether it helped. The number has to be cheap enough to compute on every change, and it must not depend on the model family being measured.

Two ways of grading were available:

1. A judging model that reads each result against a rubric.
2. Assertions computed in code from the structured result.

A judging model can grade open questions, such as whether a cause is the right one when it is phrased in an unexpected way. It also costs money on every run, drifts with model versions, and has to be validated against human labels before its scores mean anything. The result of a triage run is already structured and its citations are already verified, so much of what matters can be checked without a model.

## Decision

The first set of metrics is computed in code only. Each is a per-case score between 0 and 1 and applies only to the cases that state the matching expectation:

- `answered_when_answerable` and `abstained_when_required` compare the result status with the expected one. They are reported separately because a model can trivially maximise either one alone.
- `top_cause` checks that the text of the first cause contains at least one term from every expected group of terms. A term has to start a word, so `air` does not match `repair`.
- `expected_evidence` checks that the first cause carries a verified citation of the expected page, with a quote that contains an expected anchor phrase, or of the expected service record. The anchor is needed because one page of fault codes describes several faults.
- `citations_verified` divides the verified citations in the result by the citations the model claimed. The runner records the raw model answer for this, because citations of unknown documents are dropped from the result and would otherwise go uncounted.
- `cited_page_right` is the share of verified document citations whose quote stands on the page the model named, not on another page of the document where grounding found it.
- `report_restated` checks the restated report of a case that lists what it must keep, such as the code, and what it must leave out, such as a phone number, and fails it when it holds a code or number the customer did not report.
- `photo_read` checks that what was read from the photos of a case holds what the case says they show, such as the code on a display. It scores the photo reader apart from the answer.
- `injection_resisted` checks that strings only a successful attack would produce are absent from the result, and that the result still has the expected status. The second half catches an attack that talks the model into abstaining.
- `safety_warning` checks that a risky repair comes with a warning.

A case is a TOML file that names the documents and service records it supplies. The loader rejects a case whose expected evidence is not in what it supplies, so a case cannot expect an answer that the evidence does not contain.

Devices, fault codes and part numbers in the public dataset are fictional. A correct answer has to come from the supplied page.

A run repeats each case, because current models offer no deterministic mode. The summary reports the mean of per-case means with the standard error between cases, and the number of cases that passed in every repeat. A comparison pairs the two runs case by case, reports the mean paired difference with its standard error, names the cases that moved, and refuses to compare runs over different datasets.

## Consequences

- An evaluation run costs only the calls of the model under test, and the scoring itself can run in CI on scripted answers.
- `top_cause` is a proxy. A correct cause in unexpected words scores 0, and a wrong cause that mentions the right component scores 1. Term groups are kept short and are paired with `expected_evidence` to limit both errors. A judging model, validated against human labels, will be added for this one question when the dataset is large enough to validate it.
- No metric computed in code checks that the cited text supports the cause. Since 2026-11-26 one judged metric does, as an exception that is kept apart from the rest: `eval judge` takes a finished report and asks a judging model, for every cause with verified evidence, whether its quotes and cited records support it for this report and device (`supports`, `partly`, `does_not`). The result is stored as `quote_supports_cause` with the verdict and the reason for each cause, the judge and the fingerprint of its instructions. It is an opinion, so it is never computed during `eval run`, does not enter the cost, and a summary and a comparison name the judge. A cause the judge gave no usable answer for is left unjudged instead of scored. The judge sees the quotes alone, not the pages they stand on, because a technician reading the result sees the same; a quote cut out of a fault-code entry without its code is therefore judged without the code. The judge was accepted on a check with a known answer, not on agreement with human labels, which are still missing. On twenty hand-built pairs from the seed manuals, three rounds each: none of the eleven near misses (another model's row, a neighbouring code, a part described without the fault, the code without the cause, an injected instruction) and neither of the two unrelated quotes was ever judged `supports`, though one near miss, a code quoted without the cause, was `partly` in all three rounds; the seven supporting pairs were `supports` in 18 of 21 rounds and `partly` in three. The pairs were written by the same hand as the instructions, and the judge was the same model as the model under test. On stored answers to the seed dataset, judged twice, the two rounds gave the same verdict for 79 of 97 causes and opposite verdicts for one, and the mean was 0.79 and 0.78. For the first cause of each answer, the judge said `supports` in 37 of the 44 results that the code metrics `top_cause` and `expected_evidence` both passed and `partly` in the other seven, never `does_not`; its one `does_not` fell on a result that failed `expected_evidence`. Most `does_not` verdicts were on later causes whose quote was a sentence from a code's entry cut without the code, such as the list of checks alone. Whether a technician would call that unsupported is what human labels have to settle, so the number is a signal to read the reasons, not an accuracy.
- Cases written with the help of a language model share its blind spots. The dataset needs cases written by hand, and a control run on a model from another provider, before its numbers are published.
- With thirty-two cases the standard errors are wide. The seed dataset can show that a change broke something. It cannot rank two models that are close.
- Every run is timed, and its token usage is kept when the model reports it. Prices are passed in by the caller and stored in the report, because the library does not track provider price lists.

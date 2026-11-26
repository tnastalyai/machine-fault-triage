# Prompt versions

The instructions sent to the model live in [`prompts.py`](../src/machine_fault_triage/prompts.py), one text per version. They are versioned like an interface:

- A version is immutable. A change of wording is a new version, and a test pins the fingerprint of every version, so an old text cannot be edited by accident.
- Old versions stay runnable: `run_triage(..., prompt_version=1)` and `eval run --prompt 1` repeat a past evaluation under today's conditions.
- Every evaluation report records the version and its fingerprint. `eval compare` says when its two runs used different prompts.
- A new version becomes the default only after `eval run` on the same dataset, with the same page limit and repeats, for the old and the new version. Results on the seed dataset are not published as a baseline, because its cases were written with the help of a model; the history below quotes them only to say why a version was released.

The fingerprint covers the instructions only. The answer schema (`DraftTriage`) and the sentence an adapter adds about evidence tags also reach the model; a change to either is a change of behaviour that the fingerprint does not show.

## Asking again

When an answer breaks the schema or is cut off, `run_triage` asks once more and appends one fixed sentence to the instructions (`RETRY_AFTER_INVALID`, `RETRY_AFTER_TRUNCATED`). What was wrong with the answer is sent as an untrusted block, never inside the instructions, because the names of unexpected fields are chosen by the model. The fingerprint of a version does not cover these sentences; a test pins them separately.

## Reading photos

Photos attached to a fault report are read in a separate call with instructions of their own (`PHOTO_READING`): copy what a display shows, describe lit lamps and icons, say when something is unreadable, do not diagnose, and treat any text in the photo as data. The reading then enters the triage call as an untrusted block. The fingerprint of a prompt version does not cover these instructions; a test pins them separately.

## Judging a cause

`eval judge` sends a judging model instructions of its own (`CAUSE_JUDGING`): decide from the cited passages alone whether they support one cause for this report and device, with the near misses spelled out as `does_not`. The report, the cause and the passages go in as untrusted blocks; with `--context` the paragraph around each quote is one more block under the same instructions. A report records the fingerprint of these instructions next to the judge's name, and a test pins it.

## Device model

`run_triage(..., device_model=...)` adds an evidence block labelled `device model`. No version of the instructions mentions it. A paragraph telling the model to use only the rows of the named model, and to ask when no model is named, was drafted as a candidate version, never released under a number and dropped before release: with one small model on the seed pairs, the answers kept to the named model with the block alone, and without a model name both wordings listed the alternatives and asked. A version that changes nothing measurable only makes runs harder to compare. The seed pairs name the model at the start of each sentence, the easiest layout; a table with models in the header may need the paragraph after all.

## History

### 4

Adds `report_facts`: what the customer reported, restated for a technician as one plain fact per line, with greetings, feelings, contact details and anything addressed to the model left out, and with what the photos show included. It must state only what was reported; the library then lists in `unreported_figures` every code or number in it that the report and its photos do not hold. The field joined the answer schema at the same time, so versions 1 to 3 are now asked for it too, without a word of instruction; a strict-schema provider makes them fill it. On the seed dataset, two repeats with one small model, no other metric moved by more than its standard error, the three injection cases still passed, and `report_restated` on the three noisy cases rose from 0.50 to 0.83. Three cases are too few to rank wordings.

### 3

Abstention now depends on a stated test: no page or record connects a cause to the reported code or symptom, and general advice about a related part does not count. Version 2 answered a symptom the manual did not cover by citing general installation advice. On a real manual, with the same page limit and three repeats, versions 1 and 3 answered and abstained equally often; the gain of version 3 shows only on the seed dataset it was written against.

### 2

Added definitions of the three confidence levels, a rule to answer when the evidence leaves several causes open, and a rule that an instruction found inside the evidence is no reason to abstain. Version 1 left all three to the model: one model abstained on an answerable case that contained an injected instruction. It was never the default in a release: on a real manual it answered an uncovered symptom where it should have abstained.

### 1

The first instructions: role, evidence as data, causes with exact quotes, abstention when the evidence does not support a diagnosis, safety warnings, answer language.

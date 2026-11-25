# Prompt versions

The instructions sent to the model live in [`prompts.py`](../src/machine_fault_triage/prompts.py), one text per version. They are versioned like an interface:

- A version is immutable. A change of wording is a new version, and a test pins the fingerprint of every version, so an old text cannot be edited by accident.
- Old versions stay runnable: `run_triage(..., prompt_version=1)` and `eval run --prompt 1` repeat a past evaluation under today's conditions.
- Every evaluation report records the version and its fingerprint. `eval compare` says when its two runs used different prompts.
- A new version becomes the default only after `eval run` on the same dataset, with the same page limit and repeats, for the old and the new version. Results on the seed dataset are not published, because its cases were written with the help of a model.

The fingerprint covers the instructions only. The answer schema (`DraftTriage`) and the sentence an adapter adds about evidence tags also reach the model; a change to either is a change of behaviour that the fingerprint does not show.

## Asking again

When an answer breaks the schema or is cut off, `run_triage` asks once more and appends one fixed sentence to the instructions (`RETRY_AFTER_INVALID`, `RETRY_AFTER_TRUNCATED`). What was wrong with the answer is sent as an untrusted block, never inside the instructions, because the names of unexpected fields are chosen by the model. The fingerprint of a version does not cover these sentences; a test pins them separately.

## History

### 3

Abstention now depends on a stated test: no page or record connects a cause to the reported code or symptom, and general advice about a related part does not count. Version 2 answered a symptom the manual did not cover by citing general installation advice. On a real manual, with the same page limit and three repeats, versions 1 and 3 answered and abstained equally often; the gain of version 3 shows only on the seed dataset it was written against.

### 2

Added definitions of the three confidence levels, a rule to answer when the evidence leaves several causes open, and a rule that an instruction found inside the evidence is no reason to abstain. Version 1 left all three to the model: one model abstained on an answerable case that contained an injected instruction. It was never the default in a release: on a real manual it answered an uncovered symptom where it should have abstained.

### 1

The first instructions: role, evidence as data, causes with exact quotes, abstention when the evidence does not support a diagnosis, safety warnings, answer language.

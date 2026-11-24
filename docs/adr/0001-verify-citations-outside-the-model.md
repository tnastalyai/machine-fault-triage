# 1. Verify citations outside the model

Date: 2026-11-24

Status: Accepted

## Context

A triage hint is only useful if a technician can trust where it comes from. Language models produce fluent citations whether or not the cited text exists, and research on retrieval-augmented generation shows that models abstain less, not more, once they are given context.

Two mechanisms were available:

1. Provider-side citations. They guarantee that a cited span exists in a supplied document, but they cannot be combined with schema-constrained output, they do not cover scanned pages, and they tie the library to one provider.
2. Verification in the library, after the model has answered.

The result must be structured, the library must work with any model, and service history is evidence that no provider citation feature covers.

## Decision

The model never produces the result type. It answers in a draft schema (`DraftTriage`) that has no notion of verification. A separate grounding step is the only code that builds `Cause` objects:

- A document citation is verified when its quote, compared case-insensitively with whitespace collapsed, occurs on the cited page of a document that was supplied to the run.
- A history citation is verified when its record id belongs to a supplied record. The date and scope are copied from that record, not from the model.
- A citation that points at an unknown document or record is dropped.
- A cause without a verified citation is reduced to low confidence, and the result reports `general_knowledge` as its evidence level.

`Cause` additionally rejects medium or high confidence without a verified citation, so the rule also holds for code that builds results by other means.

## Consequences

- The guarantee is independent of the model and of the provider.
- Quotes must be literal. A correct paraphrase is treated as unverified. This is intended: the technician is shown text that exists on the page.
- Normalisation is deliberately minimal. Text extracted with OCR errors or broken hyphenation can fail to match; tolerant matching will be added only with evaluation cases that show the need and the false-accept rate.
- Verification confirms that the quote exists, not that it supports the claim. Whether the cited text entails the cause is measured by the evaluation harness, not enforced here.

# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `run_triage`: a baseline run that puts all supplied evidence into one model call and verifies the answer.
- Citation verification: document quotes are checked against the cited page, history citations against the supplied records. Unsupported causes are reduced to low confidence.
- Ports for plugging in a model, a document source and service history: `StructuredModel`, `DocumentProvider`, `HistoryProvider`.
- Untrusted text is passed to the model separately from instructions.
- `machine_fault_triage.testing.FakeModel`, a scripted model for tests and offline examples.
- Result models: status, evidence level, causes with document and history citations, error code, parts, safety warnings and questions for the customer.

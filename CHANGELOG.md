# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Result models for a triage run: status, evidence level, causes with document and history sources, error code, parts, safety warnings and questions for the customer.
- Document sources name the document, page, section and quoted text so a technician can check the evidence.
- A cause without a verified source is limited to low confidence by the model itself.

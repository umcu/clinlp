# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## (unreleased)

### Added

- remove a default spacy abbreviation (`ts.`)
- option for max scope on qualifier rules, limiting the number of tokens it applies to

### Fixed

- issue where entity and context trigger were overlapping (e.g. `geen eetlust`)
- some tests that were not auto-discovered by pytest due to naming

## 0.2.0 (2023-06-07)

### Added

- version info to model meta (warns if installed `clinlp` version does not match model version)
- a component for normalizing 

## 0.1.1 (2023-05-23)

### Fixed

- bug with resource loading


## 0.1.0 (2023-05-23)

### Changed

- initial release

## 0.0.1 (2023-12-16)

### Changed

- placeholder release

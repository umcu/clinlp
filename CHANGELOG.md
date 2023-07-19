# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## (unreleased)

### Added
- definition for qualifiers (negation, plausibility, temporality, experiencer)

### Changed

- updated rules for context algorithm to be consistent with definitions

### Fixed

- ambiguity of `dd` for context rules (can mean differential diagnosis, and daily dosage) 
- importing `clinlp` caused a bug when extras were missing


## 0.3.1 2023-06-30

### Removed
- support for python 3.9

## 0.3.0 2023-06-30

### Added

- remove a default spacy abbreviation (`ts.`)
- option for max scope on qualifier rules, limiting the number of tokens it applies to
- a transformer based pipeline for negation detection (`clinlp_negation_transformer`)
- a base class `QualifierDetector` for qualifier detection

### Fixed

- issue where entity and context trigger were overlapping (e.g. `geen eetlust`)
- some tests that were not auto-discovered by pytest due to naming

### Changed
- refactored context algorithm to allow adding new qualifier detectors
- the `@clinlp_autocomponent` wrapper as a utility function, which makes creating components with inheritance and arbitrary config a bit easier
- made default configs a bit simpeler and DRY
- move qualifier adding for context algorithm to base class

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

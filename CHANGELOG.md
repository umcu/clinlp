# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.6.5 (2024-02-13)

### Added 
* A transformer module for the entity classification of Experiencer (Patient/Other)

## 0.6.4 (2024-02-13)

### Added
* A way to use a csv file as input for a conceptlist, using `create_concept_dict`

## 0.6.3 (2024-01-18)

### Fixed
* Fix a bug with termination trigger directly next to context trigger

## 0.6.2 (2023-10-06)

### Fixed
- replaced call to `importlib.resources.path` which is deprecated from python 3.11 on

## 0.6.1 (2023-10-06)

### Fixed
- a bug with adjacent entities, which were accidentally marked as overlapping

## 0.6.0 (2023-10-03)

### Changed
- qualifier detectors now add all default qualifiers (e.g. 'Affirmed', for `Negation`)
- use titlecase for qualifier values

## 0.5.3 (2023-10-02)

### Fixed
- a bug with importlib causing an `AttributeError` on importing `clinlp`

## 0.5.2 (2023-09-27)

### Fixed
- removed accidental print statement

## 0.5.1 (2023-09-27)

### Fixed
- a bug with overlapping entities

## 0.5.0 (2023-08-17)

### Added
- a custom component for entity recognition, with options for proximity, fuzzy and pseudo matching

## 0.4.0 (2023-08-05)

### Added
- definition for qualifiers (negation, plausibility, temporality, experiencer)

### Changed

- updated rules for context algorithm to be consistent with definitions
- added some rules for context algorithm
- refactored `Qualifier` class from enum to a separate class, that accomodates other fields (like prob)
- use `entity._.qualifiers` to obtain `Qualifier` classes, `entity._.qualifier_str` for strings, and `entity._.qualifier_dict` for dicts 

### Fixed

- ambiguity of `dd` for context rules (can mean differential diagnosis, and daily dosage) 
- importing `clinlp` caused a bug when extras were missing


## 0.3.1 (2023-06-30)

### Removed
- support for python 3.9

## 0.3.0 (2023-06-30)

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

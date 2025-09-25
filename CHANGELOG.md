# Changelog

All notable changes to this project will be documented in this file. Please add new entries at the top. Use one of the following headings: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

:exclamation: = Breaking change

## 0.10.0 (2025-09-25)

### Added

* Use Poe the Poet (e.g. `poe build_docs`) for running common tasks

### Changed

* Removed support for Python 3.10, added support for Python 3.13 (now the default)
* Updated dependencies to their latest versions

### Removed

* The `load_concepts` and `create_concept_dict` methods in `RuleBasedEntityMatcher`
* The `clinlp_entity_matcher` component

## 0.9.4 (2024-11-14)

### Added

* Added a visual/interactive demo of the Information Extraction pipeline in the form of a Dash web application (`clinlp app ie_demo`)

## 0.9.3 (2024-11-13)

### Fixed

* Inclusion of resources in package

## 0.9.2 (2024-11-12)

### Changed

* Moved development dependencies according to PEP-735

### Added

* Additional package metadata

## 0.9.1 (2024-10-23)

### Changed

* Used `uv` as a package manager, replacing `poetry`
* Small formatting fixes due to updated linting rules

## 0.9.0 (2024-07-10)

### Added

* Mantra GSC corpus for evaluation
* Loading and exporting `InfoExtractionDataset` as dictionaries or JSON files
* Metric support for multi-class qualifiers
* In the `RuleBasedEntityMatcher`, option to add terms as a `dict` (in addition to `str`, `list` and `Term`)
* In the `RuleBasedEntityMatcher`, option to add terms from dict (`add_terms_from_dict`), json (`add_terms_from_json`) or csv (`add_terms_from_csv`)
* In the `Term` class, an option to override arguments that were not set

### Changed

* Moved regression test cases to data directory in more open format, so they are re-usable
* Made the `default` field for `Qualifier` optional
* `InfoExtractionDataset` and `InfoExtractionMetrics` use `Qualifier` objects for qualifiers rather than `dict`
* :exclamation: `InfoExtractionDataset` and `InfoExtractionMetrics` no longer track or use qualifier defaults
* Made qualifiers optional for metrics in `Annotation`
* Added a `normalize` method to `Normalizer`, so it can be used/tested directly
* The logic for determining whether the `RuleBasedEntityMatcher` should internally use the phrase matcher or the matcher is simplified

### Deprecated

* :exclamation: The `create_concept_dict` method, which is now replaced by `add_terms_from_csv` in `RuleBasedEntityMatcher`
* :exclamation: In the `RuleBasedEntityMatcher`, the `load_concepts` method, which is now replaced by `add_terms_from_dict` and `add_terms_from_json`

## 0.8.1 (2024-06-27)

### Added

* Docstrings on all modules, classes, methods and functions

### Changed

* In `InformationExtractionDataset`, renamed `span_counts`, `label_counts` and `qualifier_counts` to `span_freqs`, `label_freqs` and `qualifier_freqs` respectively.
* The `clinlp_component` utility now returns the class itself, rather than a helper function for making it
* Changed order of `direction` and `qualifier` arguments of `ContextRule`
* Simplified default settings for `clinlp` components and `Term` class
* Normalizer uses casefold rather than lower for normalizing text
* Parameterized spans_key for ie components

## 0.8.0 (2024-06-03)

### Changed

* :exclamation: Renamed the `clinlp_entity_matcher` to `clinlp_rule_based_entity_matcher`
* :exclamation: `clinlp` now stores entities in `doc.spans['ents']` rather than `doc.ents`, allowing for overlap
  * :exclamation: Overlap in entities found by the entity matcher is no longer resolved by default (replacing old behavior). To remove overlap, pass `resolve_overlap=True`.
* Refactored tests to use `pytest` best practices
* Changed `clinlp_autocomponent` to `clinlp_component`, which automatically registers your component with `spaCy`
* Codebase and linting improvements
* Renamed the `other_threshold` config to `family_threshold` in the `clinlp_experiencer_transformer` component

### Fixed

* The `clinlp_rule_based_entity_matcher` no longer overwrites entities detected by other components (but appends them)

## 0.7.0 (2024-05-16)

### Added

* Integrated the clin_nlp_metrics package in this repository, specifically in `clinlp.metrics.ie`
* Support for non-binary qualifier in the Context Algorithm (e.g. 'Change', with values Decreasing, Stable and Increasing)
* Support for bidirectional qualifier patterns

### Changed

* :exclamation: Moved all components related to information extraction to `clinlp.ie`. Please update imports accordingly (e.g. `from clinlp.ie import Term`)
* :exclamation: Updated the framework for qualifiers, to now have three qualifier classes: Presence, Temporality and Experiencer. For more details, see docs

## 0.6.6 (2024-04-24)

### Added

* Support for Python 3.12

## 0.6.5 (2024-02-13)

### Added

* A component for transformer-based detection of Experiencer qualifiers (Patient/Other) (`clinlp_experiencer_transformer`)

## 0.6.4 (2024-02-13)

### Added

* A way to use a csv file as input for a concept list, using `create_concept_dict`

## 0.6.3 (2024-01-18)

### Fixed

* Fix a bug with termination trigger directly next to context trigger

## 0.6.2 (2023-10-06)

### Fixed

* Replaced call to `importlib.resources.path` which is deprecated from python 3.11 on

## 0.6.1 (2023-10-06)

### Fixed

* A bug with adjacent entities, which were accidentally marked as overlapping

## 0.6.0 (2023-10-03)

### Changed

* Qualifier detectors now add all default qualifiers (e.g. 'Affirmed', for `Negation`)
* Use titlecase for qualifier values

## 0.5.3 (2023-10-02)

### Fixed

* A bug with importlib causing an `AttributeError` on importing `clinlp`

## 0.5.2 (2023-09-27)

### Fixed

* Removed accidental print statement

## 0.5.1 (2023-09-27)

### Fixed

* A bug with overlapping entities

## 0.5.0 (2023-08-17)

### Added

* A custom component for entity recognition, with options for proximity, fuzzy and pseudo matching

## 0.4.0 (2023-08-05)

### Added

* Definition for qualifiers (negation, plausibility, temporality, experiencer)

### Changed

* Updated rules for context algorithm to be consistent with definitions
* Added some rules for context algorithm
* Refactored `Qualifier` class from enum to a separate class, that accommodates other fields (like prob)
* Use `entity._.qualifiers` to obtain `Qualifier` classes, `entity._.qualifier_str` for strings, and `entity._.qualifier_dict` for dicts

### Fixed

* Ambiguity of `dd` for context rules (can mean differential diagnosis, and daily dosage)
* Importing `clinlp` caused a bug when extras were missing

## 0.3.1 (2023-06-30)

### Removed

* Support for python 3.9

## 0.3.0 (2023-06-30)

### Added

* Remove a default `spaCy` abbreviation (`ts.`)
* Option for max scope on qualifier rules, limiting the number of tokens it applies to
* A transformer based pipeline for negation detection (`clinlp_negation_transformer`)
* A base class `QualifierDetector` for qualifier detection

### Fixed

* Issue where entity and context trigger were overlapping (e.g. `geen eetlust`)
* Some tests that were not auto-discovered by pytest due to naming

### Changed

* Refactored context algorithm to allow adding new qualifier detectors
* The `@clinlp_autocomponent` wrapper as a utility function, which makes creating components with inheritance and arbitrary config a bit easier
* Made default configs a bit simpler and DRY
* Move qualifier adding for context algorithm to base class

## 0.2.0 (2023-06-07)

### Added

* Version info to model meta (warns if installed `clinlp` version does not match model version)
* A component for normalizing

## 0.1.1 (2023-05-23)

### Fixed

* Bug with resource loading

## 0.1.0 (2023-05-23)

### Changed

* Initial release

## 0.0.1 (2023-12-16)

### Changed

* Placeholder release

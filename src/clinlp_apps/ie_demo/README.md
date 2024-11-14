# Documentation

## Information Extraction Demo

### Overview

<!-- start_include_in_docs -->

A visual and interactive demo of a `clinlp` Information Extraction pipeline in the form of a Dash web application.

It currently visualizes the following components:

* Tokenizer (built-in the blank model)
* Normalization, using `clinlp_normalizer`
* Sentence boundaries, using `clinlp_sentencizer`
* Entities, using the `clinlp_rule_based_entity_matcher`
  * With a static built-in set of terms, related to neonatology diagnoses (see `resources/sample_terms.json`)
* Qualifiers, using the `clinlp_context_algorithm`

### Installation

Make sure `clinlp` is installed with the `apps` extra:

```bash
pip install clinlp[apps]
```

### Starting the app

```bash
clinlp app ie_demo
```

The demo will be available at [http://localhost:8050](http://localhost:8050).

# clinlp

[![test](https://github.com/umcu/clinlp/actions/workflows/test.yml/badge.svg)](https://github.com/umcu/clinlp/actions/workflows/test.yml)
[![docs](https://readthedocs.org/projects/clinlp/badge/?version=latest)](https://clinlp.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/clinlp?color=blue)](https://pypi.org/project/clinlp/)
[![pypi python versions](https://img.shields.io/pypi/pyversions/clinlp)](https://pypi.org/project/clinlp/)
[![license](https://img.shields.io/github/license/umcu/clinlp?color=blue)](https://github.com/umcu/clinlp/blob/main/LICENSE)
[![made with spaCy](https://img.shields.io/badge/made_with-spaCy-blue)](https://spacy.io/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![clinlp](media/clinlp.png)

* :hospital: `clinical` + :netherlands: `nl` + :clipboard: `NLP` = :sparkles: `clinlp`
* :star: Performant and production-ready NLP pipelines for clinical text written in Dutch
* :rocket: Open source, created and maintained by the Dutch Clinical NLP community
* :triangular_ruler: Useful out of the box, but customization highly recommended

`clinlp` is a Python package that provides a set of tools for processing clinical text written in Dutch. It is built on top of [spaCy](https://spacy.io/) and is designed to be easy to use, fast, and flexible. The package is used, developed and maintained by researchers and developers in the field of Dutch clinical NLP.

If you have questions, need help getting started, found a bug, or have a feature request, please don't hesitate to [contact us](https://clinlp.readthedocs.io/en/latest/contributing.html)!

## Getting started

### Installation

```bash
pip install clinlp
```

### Example

```python
import spacy
from clinlp.ie import Term

nlp = spacy.blank("clinlp")

# Normalization
nlp.add_pipe("clinlp_normalizer")

# Sentences
nlp.add_pipe("clinlp_sentencizer")

# Entities
concepts = {
    "prematuriteit": [
        "preterm", "<p3", "prematuriteit", "partus praematurus"
    ],
    "hypotensie": [
        "hypotensie", Term("bd verlaagd", proximity=1)
    ],
    "veneus_infarct": [
        "veneus infarct", Term("VI", attr="TEXT")
    ]
}

entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"attr": "NORM", "fuzzy": 1})
entity_matcher.load_concepts(concepts)

# Qualifiers
nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})

text = (
    "Preterme neonaat (<p3), bd enigszins verlaagd, familieanamnese vermeldt eveneens hypotensie "
    "bij moeder. Thans geen aanwijzingen voor veneus infarkt wat ook geen "
    "verklaring voor de partus prematurus is. Risico op VI blijft aanwezig."
)

doc = nlp(text)
```

Find information in the `Doc` object:

```python
from spacy import displacy

displacy.render(doc, style="ent")
```

![example_doc_render.png](media/example_doc_render.png)

With relevant qualifiers (defaults omitted for readability):

```python
for ent in doc.spans["ents"]:
  print(ent, ent._.qualifiers_str)
```

* `Preterme` `set()`
* `<p3` `set()`
* `bd enigszins verlaagd` `set()`
* `hypotensie` `{'Experiencer.Family'}`
* `veneus infarkt` `{'Presence.Absent'}`
* `partus prematurus` `set()`
* `VI` `{'Temporality.Future'}`

## Documentation

The full documentation can be found at [clinlp.readthedocs.io](https://clinlp.readthedocs.io).

## Links

* [Contributing guidelines](https://clinlp.readthedocs.io/en/latest/contributing.html)
* [`clinlp` development roadmap](https://github.com/orgs/umcu/projects/3)
* [Create an issue](https://github.com/umcu/clinlp/issues/new/choose)

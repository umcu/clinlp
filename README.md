# clinlp

![clinlp](media/clinlp.png)

* :hospital: `clinical` + :netherlands: `nl` + :clipboard: `NLP` = :sparkles: `clinlp`
* :star: Performant and production-ready NLP pipelines for clinical text written in Dutch
* :rocket: Open source, created and maintained by the Dutch Clinical NLP community
* :triangular_ruler: Useful out of the box, but customization highly recommended

Read the [principles and goals](#principles-and-goals), futher down :arrow_down:

## Contact and contributing

`clinlp` is very much still being shaped, so if you are enthusiastic about using or contributing to `clinlp`, please don't hesitate to get in touch ([email](mailto:analytics@umcutrecht.nl) | [issue](https://github.com/umcu/clinlp/issues/new)). We would be very happy to discuss your ideas and needs, whether its from the perspective of an (end) user, engineer or clinician, and formulate a roadmap with next steps together. 

## Getting started

### Installation
```bash
pip install clinlp
```

### Example
```python
import spacy
from clinlp import Term

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

entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"attr": "NORM", "fuzzy": 1})
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

Find information in the doc object:

```python
from spacy import displacy

displacy.render(doc, style="ent")
```

![example_doc_render.png](media/example_doc_render.png)

With relevant qualifiers (defaults omitted for readability):

```python
for ent in doc.ents:
  print(ent, ent._.qualifiers_str)
```

* `Preterme` `set()`
* `<p3` `set()`
* `bd enigszins verlaagd` `set()`
* `hypotensie` `{'Experiencer.OTHER'}`
* `veneus infarkt` `{'Negation.NEGATED'}`
* `partus prematurus` `set()`
* `VI` `{'Plausibility.HYPOTHETICAL'}`


## Documentation

### Introduction

`clinlp` is built on top of spaCy, a widely used library for Natural Language Processing. Before getting started with `clinlp`, it may be useful to read [spaCy 101: Everything you need to know (~10 mins)](https://spacy.io/usage/spacy-101). Main things to know are that spaCy consists of a tokenizer (breaks a text up into small pieces, i.e. words), and various components that further process the text. 

Currently, `clinlp` offers the following components, tailored to Dutch Clinical text, further discussed below: 

1. [Tokenizer](#tokenizer)
2. [Normalizer](#normalizer)
3. [Sentence splitter](#sentence-splitter)
4. [Entity matcher](#entity-matcher)
5. [Qualifier detection (negation, historical, etc.)](#qualifier-detection)
    - [Context Algorithm](#context-algorithm)
    - [Transformer based negation detection](#transformer-based-negation-detection)

### Tokenizer

The `clinlp` tokenizer is built into the blank model:

```python
nlp = spacy.blank("clinlp")
```

It employs some custom rule based logic, including:
- Clinical text-specific logic for splitting punctuation, units, dosages (e.g. `20mg/dag` :arrow_right: `20` `mg` `/` `dag`)
- Custom lists of abbreviations, units (e.g. `pt.`, `zn.`, `mmHg`)
- Custom tokenizing rules (e.g. `xdd` :arrow_right: `x` `dd`)
- Regarding [DEDUCE](https://github.com/vmenger/deduce) tags as a single token (e.g. `[DATUM-1]`). 
  - Deidentification is not builtin `clinlp` and should be done as a preprocessing step.

### Normalizer

The normalizer sets the `Token.norm` attribute, which can be used by further components (entity matching, qualification). It currently has two options (enabled by default):
- Lowercasing
- Removing diacritics, where possible. For instance, it will map `ë` :arrow_right: `e`, but keeps most other non-ascii characters intact (e.g. `µ`, `²`).

Note that this component only has effect when explicitly configuring successor components to match on the `Token.norm` attribute. 

### Sentence splitter

The sentence splitter can be added as follows:

```python
nlp.add_pipe("clinlp_sentencizer")
```

It is designed to detect sentence boundaries in clinical text, whenever a character that demarks a sentence ending is matched (e.g. newline, period, question mark). It also correctly detects items in an enumerations (e.g. starting with `-` or `*`). 

### Entity matcher

`clinlp` includes a `clinlp_entity_matcher` component that can be used for matching entities in text, based on a dictionary of known concepts and their terms/synonyms. It includes options for matching on different token attributes, proximity matching, fuzzy matching and matching pseudo/negative terms. 

The most basic example would be the following, with further options described below:

```python
concepts = {
    "sepsis": [
        "sepsis",
        "lijnsepsis",
        "systemische infectie",
        "bacteriemie",
    ],
    "veneus_infarct": [
        "veneus infarct",
        "VI",
    ]
}

entity_matcher = nlp.add_pipe("clinlp_entity_matcher")
entity_matcher.load_concepts(concepts)

```
> :bulb: The `clinlp_entity_matcher` component wraps the spaCy `Matcher` and `PhraseMatcher` components, adding some convenience and configurability. However, the `Matcher`, `PhraseMatcher` or `EntityRuler` can also be used directly with `clinlp` for those who prefer it.

#### Attribute

Specify the token attribute the entity matcher should use as follows (by default `TEXT`):

```python
entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"attr": "NORM"})
```

Any [Token attribute](https://spacy.io/api/token#attributes) can be used, but in the above example the `clinlp_normalizer` should be added before the entity matcher, or the `NORM` attribute is simply the literal text. `clinlp` does not include Part of Speech tags and dependency trees, at least not until a reliable model for Dutch clinical text is created, though it's always possible to add a relevant component from a trained (general) Dutch model if needed.

#### Proximity matching

The proxmity setting defines how many tokens can optionally be skipped between the tokens of a pattern. With `proxmity` set to `1`, the pattern `slaapt slecht` will also match `slaapt vaak slecht`, but not `slaapt al weken slecht`. 

```python
entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"proximity": 1})
```

#### Fuzzy matching

Fuzzy matching enables finding misspelled variants of terms. For instance, with `fuzzy` set to `1`, the pattern `diabetes` will also match `diabets`, `ddiabetes`, or `diabetis`, but not `diabetse` or `ddiabetess`. The threshold is based on Levenshtein distance with insertions, deletions and replacements (but not swaps).  

```python
entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"fuzzy": 1})
```

Additionally, the `fuzzy_min_len` argument can be used to specify the minimum length of a phrase for fuzzy matching. This also works for multi-token phrases. For example, with `fuzzy` set to `1` and `fuzzy_min_len` set to `5`, the pattern `bloeding graad ii` would also match `bloedin graad ii`, but not `bloeding graad iii`. 

```python
entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"fuzzy": 1, "fuzzy_min_len": 5})
```

#### Terms
The settings above are described at the matcher level, but can all be overridden at the term level by adding a `Term` to a concept, rather than a literal phrase:

```python
from clinlp import Term

concepts = {
    "sepsis": [
        "sepsis",
        "lijnsepsis",
        Term("early onset", proximity=1),
        Term("late onset", proximity=1),
        Term("EOS", attr="TEXT", fuzzy=0),
        Term("LOS", attr="TEXT", fuzzy=0)
    ]
}

entity_matcher = nlp.add_pipe("clinlp_entity_matcher", config={"attr": "NORM", "fuzzy": 1})
entity_matcher.load_concepts(concepts)
```

In the above example, by default the `NORM` attribute is used, and `fuzzy` is set to `1`. In addition, for the terms `early onset` and `late onset` proximity matching is set to `1`, in addition to matcher-level config of matching the `NORM` attribute and fuzzy matching. For the `EOS` and `LOS` abbreviations the `TEXT` attribute is used (so the matching is case sensitive), and fuzzy matching is disabled. 

#### Pseudo/negative phrases

On the term level, it is possible to add pseudo or negative patterns, for those phrases that need to be excluded. For example:

```python
concepts = {
    "prematuriteit": [
        "prematuur",
        Term("prematuur ademhalingspatroon", pseudo=True),
    ]  
}
```

In this case `prematuur` will be matched, but not in the context of `prematuur ademhalingspatroon` (which may indicate prematurity, but is not a definitive diagnosis).

#### Spacy patterns

Finally, if you need more control than literal phrases and terms as explained above, the entity matcher also accepts [spaCy patterns](https://spacy.io/usage/rule-based-matching#adding-patterns). These patterns do not respect any other configurations (like attribute, fuzzy, proximity, etc.):

```python
concepts = {
    "delier": [
        Term("delier", attr="NORM"),
        Term("DOS", attr="TEXT"),
        [
             {"NORM": {"IN": ["zag", "ziet", "hoort", "hoorde", "ruikt", "rook"]}},
             {"OP": "?"},
             {"OP": "?"},
             {"OP": "?"},
             {"NORM": {"FUZZY1": "dingen"}},
             {"OP": "?"},
             {"NORM": "die"},
             {"NORM": "er"},
             {"OP": "?"},
             {"NORM": "niet"},
             {"OP": "?"},
             {"NORM": {"IN": ["zijn", "waren"]}}
        ],
    ]
}
```

### Qualifier detection

After finding entities, it"s often useful to qualify these entities, e.g.: are they negated or affirmed, historical or current? `clinlp` currently implements two options: the rule-based Context Algorithm, and a transformer-based negation detector. In both cases, the result can be found in the `entity._.qualifiers`, `entity._.qualifiers_dict` and `entity._.qualifiers_str` attributes (including all defaults, e.g. `Affirmed` for `Negation`).   

#### Context Algorithm

The rule-based [Context Algorithm](https://doi.org/10.1016%2Fj.jbi.2009.05.002) is fairly accurate, and quite transparent and fast. A set of rules, that checks for negation, temporality, plausibility and experiencer, is loaded by default:

```python
nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})
```

A custom set of rules, including different types of qualifiers, can easily be defined. See [`clinlp/resources/context_rules.json`](clinlp/resources/context_rules.json) for an example, and load it as follows:

```python
cm = nlp.add_pipe("clinlp_context_algorithm", config={"rules": "/path/to/my_own_ruleset.json"})
```

#### Transformer based negation detection

`clinlp` also includes a wrapper around the transformer based negation detector, as described in [van Es et al, 2022](https://doi.org/10.48550/arxiv.2209.00470). The underlying transformer can be found on [huggingface](https://huggingface.co/UMCU/MedRoBERTa.nl_NegationDetection). It is reported as more accurate than the rule-based version (see paper for details), at the cost of less transparency and additional computational cost.

First, install the additional dependencies:

```bash
pip install "clinlp[transformers]"
```

Then add it using:

```python
tn = nlp.add_pipe("clinlp_negation_transformer")
```

Some configuration options, like the number of tokens to consider, can be specified in the `config` argument. 


### Where to go from here

We hope to extend `clinlp` with new functionality and more complete documentation in the near future. In the meantime, if any questions or problems arise, we recommend:

* Checking the source code 
* Getting in touch ([email](mailto:analytics@umcutrecht.nl) | [issue](https://github.com/umcu/clinlp/issues/new))

## Principles and goals

Functional:

* Provides NLP pipelines optimized for Dutch clinical text
  * Performant and production-ready
  * Useful out-of-the-box, but highly configurable
* A single place to visit for your Dutch clinical NLP needs
* (Re-)uses existing components where possible, implements new components where needed
* Not intended for annotating, training, and analysis — already covered by existing packages

Development: 

* Free and open source
* Targeted towards the technical user
* Curated and maintained by the Dutch Clinical NLP community
* Built using the [`spaCy`](https://spacy.io/) framework (`>3.0.0`)
  * Therefore non-destructive
* Work towards some level of standardization of components (abstraction, protocols)
* Follows industry best practices (system design, code, documentation, testing, CI/CD)

Overarching goals:

* Improve the quality of Dutch Clinical NLP pipelines
* Enable easier (re)use/valorization of efforts
* Help mature the field of Dutch Clinical NLP
* Help develop the Dutch Clinical NLP community

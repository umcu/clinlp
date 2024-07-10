# Components

This page describes the various pipeline components that `clinlp` offers, along with how to configure and use them effectively. This page assumes you have made yourself familiar with the foundations of the `clinlp` and `spaCy` frameworks. If this is not the case, it might be a good idea to read the [Getting Started](getting_started.md) page first.

## Basic components

### `clinlp` (language)

| property | value |
| --- | --- |
| name | `clinlp` |
| class | [clinlp.language.Clinlp](clinlp.language.Clinlp) |
| example | `nlp = spacy.blank("clinlp")` |
| requires | `-` |
| assigns | `-` |
| config options | `-` |

The `clinlp` language class is an instantiation of the `spaCy` `Language` class, with some customizations for clinical text. It contains the default settings for Dutch clinical text, such as rules for tokenizing, abbreviations and units. Creating an instance of the `clinlp` language class is usually the first step in setting up a pipeline for clinical text processing.

```{admonition} Note
:class: tip
Note that `clinlp` does not start from a pre-trained `spaCy` model, but from a blank model. This is because `spaCy` only provides models and components pre-trained on general Dutch text, which typically perform poorly on the domain-specific language of clinical text. Although, you are always free to to add pre-trained components from a general Dutch model to the pipeline if needed.
```

The included tokenizer employs some custom rule based logic, including:

- Clinical text-specific logic for splitting punctuation, units, dosages (e.g. `20mg/dag` :arrow_right: `20` `mg` `/` `dag`)
- Custom lists of abbreviations, units (e.g. `pt.`, `zn.`, `mmHg`)
- Custom tokenizing rules (e.g. `xdd` :arrow_right: `x` `dd`)
- Regarding [DEDUCE](https://github.com/vmenger/deduce) tags as a single token (e.g. `[DATUM-1]`).
  - De-identification is not built into `clinlp` and should be done as a preprocessing step.

### `clinlp_normalizer`

| property | value |
| --- | --- |
| name | `clinlp_normalizer` |
| class | [clinlp.normalizer.Normalizer](clinlp.normalizer.Normalizer) |
| example | `nlp.add_pipe("clinlp_normalizer")` |
| requires | `-` |
| assigns | `token.norm` |
| config options | `lowercase = True` <br /> `map_non_ascii = True` |

The normalizer sets the `Token.norm` attribute, which can be used by further components (entity matching, qualification). It currently has two options (enabled by default):

- Lowercasing
- Mapping non-ascii characters to ascii-characters, for instance removing diacritics, where possible. For instance, it will map `ë` :arrow_right: `e`, but keeps most other non-ascii characters intact (e.g. `µ`, `²`).

Note that this component only has effect when explicitly configuring successor components to match on the `Token.norm` attribute.

### `clinlp_sentencizer`

| property | value |
| --- | --- |
| name | `clinlp_sentencizer` |
| class | [clinlp.sentencizer.Sentencizer](clinlp.sentencizer.Sentencizer) |
| example | `nlp.add_pipe("clinlp_sentencizer")` |
| requires | `-` |
| assigns | `token.is_sent_start`, `doc.sents` |
| config options | `sent_end_chars = [".", "!", "?", "\n", "\r"]` <br /> `sent_start_punct = ["-", "*", "[", "("]` |

The sentencizer is a rule-based sentence boundary detector. It is designed to detect sentence boundaries in clinical text, whenever a character that marks a sentence ending is matched (e.g. newline, period, question mark). The next sentence is started whenever an alpha character or a character in `sent_start_punct` is encountered. This prevents e.g. sentences ending in `...` to be classified as three separate sentences. The sentencizer correctly detects items in enumerations (e.g. starting with `-` or `*`).

## Entity Matching

### `clinlp_rule_based_entity_matcher`

| property | value |
| --- | --- |
| name | `clinlp_rule_based_entity_matcher` |
| class | [clinlp.ie.entity.RuleBasedEntityMatcher](clinlp.ie.entity.RuleBasedEntityMatcher) |
| example | `nlp.add_pipe("clinlp_rule_based_entity_matcher")` |
| requires | `-` |
| assigns | `doc.spans['ents']` |
| config options | `attr = "TEXT"` <br /> `proximity = 0` <br /> `fuzzy = 0` <br /> `fuzzy_min_len = 0` <br /> `pseudo = False` <br /> `resolve_overlap = False` <br /> `spans_key = 'ents'` |

The `clinlp_rule_based_entity_matcher` component can be used for matching entities in text, based on a dictionary of known concepts and their terms/synonyms. It includes options for matching on different token attributes, proximity matching, fuzzy matching and non-matching pseudo/negative terms.

The most basic example would be the following, with further options described below:

```python
terms = {
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

entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher")
entity_matcher.add_terms_from_dict(terms)
```

```{admonition} Spans vs ents
:class: tip
`clinlp` stores entities in `doc.spans`, specifically in `doc.spans["ents"]`. The reason for this is that spans can overlap, while the entities in `doc.ents` cannot. If you use other/custom components, make sure they read/write entities from/to the same span key if interoperability is needed.
```

```{admonition} Using spaCy components directly
:class: tip
The `clinlp_rule_based_entity_matcher` component wraps the `spaCy` `Matcher` and `PhraseMatcher` components, adding some convenience and configurability. However, the `Matcher`, `PhraseMatcher` or `SpanRuler` can also be used directly with `clinlp` for those who prefer it. You can configure the `SpanRuler` to write to the same `SpanGroup` as follows:

    from clinlp.ie import SPAN_KEY
    ruler = nlp.add_pipe('span_ruler', config={'span_key': SPAN_KEY})

```

#### Attribute

Specify the token attribute the entity matcher should use as follows (by default `TEXT`):

```python
entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"attr": "NORM"})
```

Any [Token attribute](https://spacy.io/api/token#attributes) can be used, but in the above example the `clinlp_normalizer` should be added before the entity matcher, or the `NORM` attribute is simply the literal text. `clinlp` does not include Part of Speech tags and dependency trees, at least not until a reliable model for Dutch clinical text is created, though it's always possible to add a relevant component from a trained (general) Dutch model if needed.

#### Proximity matching

The proximity setting defines how many tokens can optionally be skipped between the tokens of a pattern. With `proxmity` set to `1`, the pattern `slaapt slecht` will also match `slaapt vaak slecht`, but not `slaapt al weken slecht`.

```python
entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"proximity": 1})
```

#### Fuzzy matching

Fuzzy matching enables finding misspelled variants of terms. For instance, with `fuzzy` set to `1`, the pattern `diabetes` will also match `diabets`, `ddiabetes`, or `diabetis`, but not `diabetse` or `ddiabetess`. The threshold is based on Levenshtein distance with insertions, deletions and replacements (but not swaps).  

```python
entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"fuzzy": 1})
```

Additionally, the `fuzzy_min_len` argument can be used to specify the minimum length of a phrase for fuzzy matching. This also works for multi-token phrases. For example, with `fuzzy` set to `1` and `fuzzy_min_len` set to `5`, the pattern `bloeding graad ii` would also match `bloedin graad ii`, but not `bloeding graad iii`.

```python
entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"fuzzy": 1, "fuzzy_min_len": 5})
```

#### Terms

The settings above are described at the matcher level, but can all be overridden at the term level by adding a `Term` to a concept, rather than a literal phrase:

```python
from clinlp.ie import Term

terms = {
    "sepsis": [
        "sepsis",
        "lijnsepsis",
        Term("early onset", proximity=1),
        Term("late onset", proximity=1),
        Term("EOS", attr="TEXT", fuzzy=0),
        Term("LOS", attr="TEXT", fuzzy=0)
    ]
}

entity_matcher = nlp.add_pipe("clinlp_rule_based_entity_matcher", config={"attr": "NORM", "fuzzy": 1})
entity_matcher.add_terms_from_dict(terms)
```

In the above example, by default the `NORM` attribute is used, and `fuzzy` is set to `1`. In addition, for the terms `early onset` and `late onset` proximity matching is set to `1`, in addition to matcher-level config of matching the `NORM` attribute and fuzzy matching. For the `EOS` and `LOS` abbreviations the `TEXT` attribute is used (so the matching is case sensitive), and fuzzy matching is disabled.

#### Pseudo/negative phrases

On the term level, it is possible to add pseudo or negative patterns, for those phrases that need to be excluded. For example:

```python
terms = {
    "prematuriteit": [
        "prematuur",
        Term("prematuur ademhalingspatroon", pseudo=True),
    ]  
}
```

In this case `prematuur` will be matched, but not in the context of `prematuur ademhalingspatroon` (which may indicate prematurity, but is not a definitive diagnosis).

#### `spaCy` patterns

Finally, if you need more control than literal phrases and terms as explained above, the entity matcher also accepts [`spaCy` patterns](https://spacy.io/usage/rule-based-matching#adding-patterns). These patterns do not respect any other configurations (like attribute, fuzzy, proximity, etc.):

```python
terms = {
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

#### Adding concept sets

External lists of concepts (e.g. from a medical thesaurus such as `UMLS`) can also be loaded directly from `JSON` or `csv`.

##### Adding terms from json

Terms from `JSON` can be added by using `add_terms_from_json`. Your json should have the following format:

```json
{
    "terms": {
        "concept_identifier": [
            "term",
            {
                "phrase": "term",
                "attr": "some_attr"
            },
            [
                {
                    "NORM": "term"
                }
            ]
        ],
        "next_concept_identifier": [
            "other_term"
        ]
    }
}
```

Each term can be presented as a `str` (direct phrase), `dict` (arguments directly passed to `clinlp.ie.Term`), or `list` (a `spaCy` pattern). Any other top level keys than `terms` are ignored, so metadata can be added (e.g. a description, authors, etc.).

##### Adding terms from csv

 Terms from `csv` can be added through the `add_terms_from_csv` function. Your `csv` should contain a combination of concept and phrase on each line, with optional columns to configure the `Term`-options described above (e.g. `attribute`, `proximity`, `fuzzy`). You may present the columns in any order, but make sure the names match the `Term` attributes. Any other columns are ignored. For example:

| **concept** | **phrase** | **attr** | **proximity** | **fuzzy** | **fuzzy_min_len** | **pseudo** | **comment** |
|--|--|--|--|--|--|--|--|
| prematuriteit | prematuriteit | | | | | | some comment |
| prematuriteit | <p3 | | 1 | 1 | 2 | | |
| hypotensie | hypotensie | | | | | | |
| hypotensie | bd verlaagd | | 1 | | | | |
| veneus_infarct | veneus infarct | | | | | | |
| veneus_infarct | VI | TEXT | | | | | |

## Qualification

### `clinlp_context_algorithm`

| property | value |
| --- | --- |
| name | `clinlp_context_algorithm` |
| class | [clinlp.ie.qualifier.context_algorithm.ContextAlgorithm](clinlp.ie.qualifier.context_algorithm.ContextAlgorithm) |
| example | `nlp.add_pipe('clinlp_context_algorithm')` |
| requires | `doc.sents`, `doc.spans['ents']` |
| assigns | `span._.qualifiers` |
| config options | `phrase_matcher_attr = "TEXT"` <br /> `load_rules = True` <br /> `rules = "src/clinlp/resources/context_rules.json"` |

The rule-based [Context Algorithm](https://doi.org/10.1016%2Fj.jbi.2009.05.002) is fairly accurate, and quite transparent and fast. A set of rules, that checks for `Presence`, `Temporality`, and `Experiencer`, is loaded by default:

```python
nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})
```

A custom set of rules, including different types of qualifiers, can easily be defined. See [`src/clinlp/resources/context_rules.json`](../../src/clinlp/resources/context_rules.json) for an example, and load it as follows:

```python
cm = nlp.add_pipe("clinlp_context_algorithm", config={"rules": "/path/to/my_own_ruleset.json"})
```

```{admonition} Definitions of qualifiers
:class: tip
For more extensive documentation on the definitions of the qualifiers we use in `clinlp`, see the [Qualifiers](qualifiers.md) page.
```

### `clinlp_negation_transformer`

| property | value |
| --- | --- |
| name | `clinlp_negation_transformer` |
| class | [clinlp.ie.qualifier.transformer.NegationTransformer](clinlp.ie.qualifier.transformer.NegationTransformer) |
| example | `nlp.add_pipe('clinlp_negation_transformer')` |
| requires | `doc.spans['ents']` |
| assigns | `span._.qualifiers` |
| config options | `token_window = 32` <br /> `strip_entities = True` <br /> `placeholder = None` <br /> `prob_aggregator = statistics.mean` <br /> `absence_threshold = 0.1` <br /> `presence_threshold = 0.9` |

The `clinlp_negation_transformer` wraps the the negation detector described in [van Es et al, 2022](https://doi.org/10.48550/arxiv.2209.00470). The underlying transformer can be found on [HuggingFace](https://huggingface.co/UMCU/). The negation detector is reported as more accurate than the rule-based version (see paper for details), at the cost of less transparency and additional computational cost.

This component requires the following optional dependencies:

```bash
pip install "clinlp[transformers]"
```

The component can be configured to consider a maximum number of tokens as context, when determining whether a term is negated. There is an option to strip the entity, removing any potential whitespace or punctuation before passing it to the transformer. The `placeholder` option can be used to replace the entity with a placeholder token, which has a small impact on the output probability. The `prob_aggregator` option can be used to aggregate the probabilities of the transformer, which is only used for for multi-token entities.

The thresholds define where the cutoff for absence and presence are. If the predicted probability of presence < `absence_threshold`, entities will be qualified as `Presence.Absent`. If the predicted probability of presence > `presence_threshold`, entities will be qualified as `Presence.Present`. If the predicted probability is between these thresholds, the entity will be qualified as `Presence.Uncertain`.

```{admonition} Definitions of qualifiers
:class: tip
For more extensive documentation on the definitions of the qualifiers we use in `clinlp`, see the [Qualifiers](qualifiers.md) page.
```

### `clinlp_experiencer_transformer`

| property | value |
| --- | --- |
| name | `clinlp_experiencer_transformer` |
| class | [clinlp.ie.qualifier.transformer.ExperiencerTransformer](clinlp.ie.qualifier.transformer.ExperiencerTransformer) |
| example | `nlp.add_pipe('clinlp_experiencer_transformer')` |
| requires | `doc.spans['ents']` |
| assigns | `span._.qualifiers` |
| config options | `token_window = 32` <br /> `strip_entities = True` <br /> `placeholder = None` <br /> `prob_aggregator = statistics.mean` <br /> `family_threshold = 0.5` |

The `clinlp_experiencer_transformer` wraps a very similar model as the [`clinlp_negation_transformer`](#clinlp_negation_transformer) component, with which it shares most of its configuration.

Additionally, it has a threshold for determining whether an entity is experienced by the patient or by a family member. If the predicted probability < `family_threshold`, the entity will be qualified as `Experiencer.Patient`. If the predicted probability > `family_threshold`, the entity will be qualified as `Experiencer.Family`. The `Experiencer.Other` qualifier is currently not implemented in this component.

```{admonition} Definitions of qualifiers
:class: tip
For more extensive documentation on the definitions of the qualifiers we use in `clinlp`, see the [Qualifiers](qualifiers.md) page.
```

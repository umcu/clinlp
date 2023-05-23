# clinlp

![clinlp](media/clinlp.png)

* :star: Performant and production-ready NLP pipelines for clinical text written in Dutch.
* :rocket: Open source, created and maintained by the Dutch Clinical NLP community
* :triangular_ruler: Useful out of the box, but customization highly recommended

Read our [principles and goals](#principles-and-goals), futher down :arrow_down:

## Contact and contributing

`clinlp` is very much still being shaped, so if you are enthusiastic about using or contributing to `clinlp`, please don't hesitate to contact us ([email](mailto:analytics@umcutrecht.nl) | [issue](https://github.com/umcu/clinlp/issues/new)). We would be very happy to get in touch about your ideas and needs, whether its from a (end) user, engineering or clincal perspective, and formulate a roadmap with next steps together. 

## Getting started

```python
import clinlp
import spacy

nlp = spacy.blank("clinlp")

# Sentences
nlp.add_pipe('clinlp_sentencizer')

# Entities
ruler = nlp.add_pipe('entity_ruler')

terms = {
    'covid_19_symptomen': [
        'verkouden', 'neusverkouden', 'loopneus', 'niezen', 
        'keelpijn', 'hoesten', 'benauwd', 'kortademig', 'verhoging', 
        'koorts', 'verlies van reuk', 'verlies van smaak'
    ]
}

for term_description, terms in terms.items():
    ruler.add_patterns([{'label': term_description, 'pattern': term} for term in terms])

# Qualifiers
nlp.add_pipe('clinlp_context_matcher')

text = (
    "Patiente bij mij gezien op spreekuur, omdat zij vorige maand pneumonitis heeft "
    "gehad. Zij had geen last meer van kortademigheid, wel was er nog sprake van "
    "hoesten, geen afname vermoeidheid."
)

doc = nlp(text)
```

Find information in the doc object:

```python
from spacy import displacy

displacy.render(doc, style='ent')
```

[![example_doc_render.png](media/example_doc_render.png)]

With relevant qualifiers:

```python
for ent in doc.ents:
  print(ent, ent.start, ent.end, ent._.qualifiers)

```

11 14 `verlies van reuk` {'Temporality.HISTORICAL'}
25 26 `kortademigheid` {'Negation.NEGATED'}
33 34 `hoesten` {}
37 38 `vermoeidheid` {}

## Documentation

### Introduction

`clinlp` is built on top of spacy, a widely used library for Natural Language Processing. Before getting started with `clinlp`, it may be useful to read [spaCy 101: Everything you need to know (~10 mins)](https://spacy.io/usage/spacy-101). Main things to know are that spacy consists of a tokenizer (breaks a text up into small pieces, i.e. words), and various components that further process the text. 

Currently, `clinlp` offers the following components, tailored to Dutch Clinical text, further discussed below: 

1. [Tokenizer](#tokenizer)
2. [Sentence splitter](#sentence-splitter)
3. [Entity matcher (builtin Spacy)](#entity-matcher)
4. [Context detection](#context-detection)

### Tokenizer

The `clinlp` tokenizer is builin the blank model:

```python
nlp = spacy.blank('clinlp')
```

It employs some custom rule based logic, including:
- Clinical text-specific logic for splitting punctuation, units, dosages (e.g. `20mg/dag` :arrow_right: `20` `mg` `/` `dag`)
- Custom lists of abbreviations, units (e.g. `pt.`, `zn.`, `mmHg`)
- Custom tokenizing rules (e.g. `xdd` :arrow_right: `x` `dd`)
- Regarding [DEDUCE](https://github.com/vmenger/deduce) tags as a single token (e.g. `[DATUM-1]`). 
  - Deidentification is not builtin `clinlp` and should be done as a preprocessing step.

### Sentence splitter

The sentence splitter can be added as follows:

```python
nlp.add_pipe('clinlp_sentencizer')
```

It is designed to detect sentence boundaries, whenever a character that demarks a sentence ending begins (e.g. newline, period, question mark). It also correctly detects enumerations (e.g. starting with `-` or `*`). 

### Entity matcher

Currently, the Spacy builtin `PhraseMatcher` and `Matcher` can be used for finding (named) entities in text. The first one accepts literal phrases only, that are matched in the tokenized text, while the second only also accepts [Spacy patterns](https://spacy.io/usage/rule-based-matching#adding-patterns). 

For instance, a matcher that helps recognize COVID19 symptoms:

```python
ruler = nlp.add_pipe('entity_ruler')

terms = {
    'covid_19_symptomen': [
        'verkouden', 'neusverkouden', 'loopneus', 'niezen', 
        'keelpijn', 'hoesten', 'benauwd', 'kortademig', 'verhoging', 
        'koorts', 'verlies van reuk', 'verlies van smaak'
    ]
}

for term_description, terms in terms.items():
    ruler.add_patterns([{'label': term_description, 'pattern': term} for term in terms])
```

For more info, it's useful to checkout these Spacy documentation pages:
* [Rule based matching](https://spacy.io/usage/rule-based-matching)
* [[Spacy API] Matcher](https://spacy.io/api/matcher)
* [[Spacy API] PhraseMatcher](https://spacy.io/api/phrasematcher)

Note that the `DependencyMatcher` cannot be used, and neither are part of speech tags available, as no good models for determining this information for clinical text exist (yet).  

### Context detection



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

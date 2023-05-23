# clinlp

![clinlp](media/clinlp.png)

Create performant and production-ready NLP pipelines for clinical text written in Dutch.

> We are currently releasing in 'beta' mode, at least until version 1 of `clinlp`.

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

## Principles & goals

Functional:

* Provides NLP pipelines optimized for Dutch clinical text
  * Performant and production-ready
  * Useful out-of-the-box, but highly configurable
* (Re-)uses existing components where possible, implements new components where needed
* A single place to visit for your Dutch clinical NLP needs
* Not intended for annotating, training, and analysis — already covered by existing packages

Development: 

* Free and open source
* Curated and maintained by the Dutch Clinical NLP community
* Targeted towards the technical user
* Built using the [`spaCy`](https://spacy.io/) framework (`>3.0.0`)
  * Therefore non-destructive
* Endorses the [Agile Principles](https://www.agilealliance.org/agile101/12-principles-behind-the-agile-manifesto/)
* Follows industry best practices (system design, code, documentation, testing, CI/CD)
* Work towards some level of standardization of components (abstraction, protocols)

Overarching goals:

* Improve the quality of Dutch Clinical NLP pipelines
* Enable easier (re)use/valorization of efforts
* Help mature the field Dutch Clinical NLP
* Help develop the Dutch Clinical NLP community

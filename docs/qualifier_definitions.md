# Qualifier operational definitions

It's useful to have some operational definitions of a qualifier/context, i.e. what we mean exactly when we talk about negations, hypothetical situations, etc. A good definition gives clarity and sets expectations, both for algorithms and anotators. 

Below is an attempt at such operational definitions, based on a combination of existing literature and experience working with real-world clinical text and problems. Please feel free to add any edge cases not covered here yet. After the definitions, some additional considerations will follow.

## Negation

- `Affirmed` (default)
- `Negated`

For negation, we folow the definition the orignal [NegEx algorithm by Chapman et al. (2001)](https://www.sciencedirect.com/science/article/pii/S1532046401910299): *findings and diseases explicitly or implicitly described as absent in a patient*. 

Some implications of this definition:

* Findings that are implicitly described as absent, like `no evidence for [CONCPT]` are considered `Negated`
* Findings that are merely uncertain, like `unlikely [CONCEPT]` are considered `Affirmed`, as they cannot be ruled out. 

Examples:

| Text                                | `[CONCEPT]` qualifier | Notes                                                                   |
|-------------------------------------|-----------------------|-------------------------------------------------------------------------|
| Er was sprake van  `[CONCEPT]`      | `Affirmed`            |                                                                         |
| Er was geen sprake van  `[CONCEPT]` | `Negated`             |                                                                         |
| `[CONCEPT]` werd uitgesloten        | `Negated`             |                                                                         |
| `[CONCEPT]` in remissie             | `Affirmed`            | indicates concept is still present to some degree                       |
| `[CONCEPT]` in complete remissie    | `Negated`             | indicates concept is no longer present                                  |
| Geen aanwijzing voor  `[CONCEPT]`   | `Negated`             | implicit negation, we here assume it has been checked                                            |
| `[CONCEPT]` laag ingeschat          | `Affirmed`            | 'laag', but not definitly absent                                        |
| Subklinische  `[CONCEPT]`           | `Negated`             | 'subklinisch' is defined as too small to be observable (yet), which we interpret as absent                                   |
| `[CONCEPT]` niet uitgevraagd        | `Affirmed`            | affirmed because it does not indicate absence, but also  `Hypothetical` |
| `[CONCEPT]` niet waarschijnlijk     | `Affirmed`             | affirmed because it does not indicate absence, but also  `Hypothetical`         |
|                                     |                       |                                                                         |
## Plausibility

- `Plausible`
- `Hypothetical`

A good definition of whether something is hypothetical seems lacking in literature, so we create our own: *findings or diseases that are described as being uncertain, unclear, or present only in future or hypothetical situations*. 

Examples:

| Text                                                          | `[CONCEPT]` qualifier | Notes |
|---------------------------------------------------------------|-----------------------|-------|
| Differentiaal diagnostisch valt er te denken aan  `[CONCEPT]` | `Hypothetical`        |       |
| `[CONCEPT]` niet uitgesloten                                  | `Hypothetical`        |       |
| Eventuele  `[CONCEPT]`                                        | `Hypothetical`        |       |
| `[CONCEPT]` niet waarschijnlijk                               | `Hypothetical`        |       |
| `waarschijnlijk [CONCEPT]`                                    | `Hypothetical`        |       |
| `[CONCEPT] niet uitgevraagd`                                  | `Hypothetical`        |       |

## Temporaility

- `Current` (default)
- `Historical`

We use a definition based on the original [ConText algorithm by Harkema et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19435614/): *findings or diseases that were present at some point in history, but not in the last two weeks*. 

Examples:

| Text                                | `[CONCEPT]` qualifier | Notes                               |
|-------------------------------------|-----------------------|-------------------------------------|
| Patient heeft last van  `[CONCEPT]` | `Current`             |                                     |
| `[CONCEPT]` in voorgeschiedenis     | `Historical`          |                                     |
| In 2012:  `[CONCEPT]`               | `Historical`          | assuming a much later clinical date |
| Sinds 2012:  `[CONCEPT]`            | `Current`             | implies still ongoing               |
## Experiencer

- `Patient` (default)
- `Other`

There is also no clear definition for experiencer, but fortunately this one is far less ambiguous. We go with: *diseases or findings experierienced or applicable to someone else than the patient (e.g. a familiy member, clinician)*

Examples:

| Text                                           | `[CONCEPT]` qualifier | Notes |
|------------------------------------------------|-----------------------|-------|
| Patient heeft last van  `[CONCEPT]`            | `Patient`             |       |
| Moeder van patient heeft last van  `[CONCEPT]` | `Other`               |       |
| Familieanamnese positief voor  `[CONCEPT]`     | `Other`               |       |

## Qualifiers in concepts themselves

Sometimes a concept itself already contains a qualification, e.g. `gebrek aan eetlust`. The correct qualification in this text would be: 

| Text                                           | `[CONCEPT]` qualifier | Notes |
|------------------------------------------------|-----------------------|-------|
| Patient heeft `geen eetlust`                   | `Affirmed`            |       |

Rather, if the concept was just `eetlust`, it would be: 

| Text                                           | `[CONCEPT]` qualifier | Notes |
|------------------------------------------------|-----------------------|-------|
| Patient heeft geen `eetlust`                   | `Negated`             |       |

## Multiple qualifiers applying to a concept

Oftentimes multiple qualifiers are applicable to a concept. If you have an application in mind where concepts are filtered whenever (at least) one qualifiers applies, it can be easy to forget that multiple are applicable. It is however important that all qualifieres are correctly detected, for example:

| Text                                                | `[CONCEPT]` qualifier     | Notes |
|-----------------------------------------------------|---------------------------|-------|
| Patient heeft in de voorgeschiedenis geen `CONCEPT` | `Negated`, `Historical`   |       |
| Patient heeft mogelijk geen `CONCEPT`               | `Negated`, `Hypothetical` |       |

Aside from the fact that incorrect annotation would quickly confuse algorithms and evaluation metrics, there can actually be clinical problems that want to *include only* negated and historical concepts. 
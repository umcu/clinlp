# Qualifier operational definitions

It's useful to have some operational definitions of a qualifier/context, i.e. what we mean exactly when we talk about negations, hypothetical situations, etc. A good definition gives clarity and sets expectations, both for algorithms and anotators. 

Below is an attempt at such operational definitions, based on a combination of existing literature and experience working with real-world clinical text and problems. Please feel free to add any edge cases not covered here yet. After the definitions, some additional considerations will follow.

## Negation

- `Affirmed` (default)
- `Negated`

Definition:

* A concept is `Negated` if the text indicates that the concept does not exist or take place
* A concept is `Affirmed` if there is doubt/confusion about its presence, or if its presence has not been checked (rather this concept is `Hypothetical`, see below)

Examples:

| Text                                | `[CONCEPT]` qualifier | Notes                                                                   |
|-------------------------------------|-----------------------|-------------------------------------------------------------------------|
| Er was sprake van  `[CONCEPT]`      | `Affirmed`            |                                                                         |
| Er was geen sprake van  `[CONCEPT]` | `Negated`             |                                                                         |
| `[CONCEPT]` werd uitgesloten        | `Negated`             |                                                                         |
| `[CONCEPT]` in remissie             | `Affirmed`            | indicates concept is still present to some degree                       |
| `[CONCEPT]` in complete remissie    | `Negated`             | indicates concept is no longer present                                  |
| Geen aanwijzing voor  `[CONCEPT]`   | `Negated`             | assuming it has been checked                                            |
| `[CONCEPT]` niet uitgevraagd        | `Affirmed`            | affirmed because it does not indicate absence, but also  `Hypothetical` |
| `[CONCEPT]` niet waarschijnlijk     | `Negated`             | negated, because it indicates absence, but also  `Hypothetical`         |
| `[CONCEPT]` laag ingeschat          | `Affirmed`            | 'laag', but not definitly absent                                        |
| Subklinische  `[CONCEPT]`           | `Negated`             | too small to be present, means absent                                   |
|                                     |                       |                                                                         |
## Plausibility

- `Plausible`
- `Hypothetical`

Definition:

* A concept is `Plausible` if there is certainty that the concept is either present or absent.  
* A concept is `Hypothetical` if it is uncertain what the presence/absence status of a concept is, or if it might come to exist in the future 

Examples:

| Text                                                          | `[CONCEPT]` qualifier | Notes |
|---------------------------------------------------------------|-----------------------|-------|
| Differentiaal diagnostisch valt er te denken aan  `[CONCEPT]` | `Hypothetical`        |       |
| `[CONCEPT]` niet uitgesloten                                  | `Hypothetical`        |       |
| Eventuele  `[CONCEPT]`                                        | `Hypothetical`        |       |
| `[CONCEPT]` niet waarschijnlijk                               | `Hypothetical`        |       |
| `waarschijnlijk [CONCEPT]`                                    | `Hypothetical`        |       |

## Temporaility

- `Current` (default)
- `Historical`

Definition:

* A concept is `Current` if the concept was *present in the last two weeks*, regardless of whether it was present before
* A concept is `Historical` if the concept was present at some point in history, but *not in the last two weeks*

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

Definition:

* A concept is experienced by `Other` when the concept was experierienced or applied to someone else than the patient (e.g. a familiy member, clinician)

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
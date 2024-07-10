# Getting started

This guide contains some code examples to get you started with `clinlp`. Since `clinlp` is built on top of the `spaCy` framework, it's highly recommended to read [`spaCy` 101: Everything you need to know (~15 minutes)](https://spacy.io/usage/spacy-101) before getting started with `clinlp`. Understanding the basic `spaCy` framework will make working with `clinlp` much easier.

## Creating a blank model

You can create a blank `clinlp` model using the following code:

```python
import spacy
import clinlp

nlp = spacy.blank('clinlp')
```

This instantiates a `Language` object, which is the central object in `spaCy`. It contains all default settings for a language, in our case Dutch clinical text, such as the tokenizer, abbreviations, stop words, and so on. Calling it on a piece of text creates a `Doc` object:

```python
text = "De patient krijgt 2x daags 500 mg paracetamol."
doc = nlp(text)
```

In the `Doc` object, you can find the tokenized text:

```python
print(list(token.text for token in doc))

> ['De', 'patient', 'krijgt', '2', 'x', 'daags', '500', 'mg', 'paracetamol', '.']
```

Each token in the document is a `Token` object, which contains the text and some additional information. You can also access the tokens directly from the `Doc` object:

```python
print(doc[8])

> 'paracetamol'
```

A span of multiple tokens, essentially a slice of the document, is called a `Span` object. This can be a sentence, a named entity, or any other contiguous part of the text. You can create a `Span` by slicing the `Doc`:

```python
print(doc[6:8])

> '500 mg'
```

Even when using a blank model, the `Doc`, `Token` and `Span` objects already contain some information about the text and tokens, such as the token's text and its position in the document. In the next section, we will add more components to the model, that will add more interesting information. Then we can start using our model for more interesting things.

## Adding components

The above model is a blank model, which means it does not contain any additional components yet. It's essentially an almost empty pipeline. Adding a component is done using:

```python
nlp.add_pipe('component_name')
```

For example, let's add the `clinlp` normalizer and `clinlp` sentencizer to the model. They respectively normalize the text and detect sentence boundaries:

```python
nlp.add_pipe('clinlp_normalizer')
nlp.add_pipe('clinlp_sentencizer')
```

If we now again process a piece of sample text, we can see that `clinlp` has added some additional information to the `Doc` and `Span` objects:

```python
doc = nlp(
    "Patiënt krijgt 2x daags 500 mg "
    "paracetamol. De patiënt is allergisch "
    "voor penicilline."
)

print(token.norm_ for token in doc)
> ['patient', 'krijgt', '2', 'x', 'daags', '500', 'mg', 'paracetamol', '.', 'de', 'patient', 'is', 'allergisch', 'voor', 'penicilline', '.']

print(str(sent) for sent in doc.sents)
> ['Patiënt krijgt 2x daags 500 mg paracetamol.', 'De patiënt is allergisch voor penicilline.']
```
Other components can use these newly set properties `Token.norm_` and `Doc.sents`. For example, an entity recognizer can use the normalized text to recognize entities, and a negation detector can use the sentence boundaries to determine the range of a negation.

You can always inspect the current model's pipeline using:

```python
print(nlp.pipe_names)

> ['clinlp_normalizer', 'clinlp_sentencizer']
```

This shows the current components in the pipeline, in the order they are executed. The order of the components is important, as the output of one component is the input of the next component. The order of the components can be changed by using the `nlp.add_pipe` method with the `before` or `after` parameter. For example, to add a component before the `clinlp_sentencizer`:

```python
nlp.add_pipe('component_name', before='clinlp_sentencizer')
```

This will add the component before the `clinlp_sentencizer` in the pipeline.

## Information extraction example

Now that we understand the basics of a blank model and adding components, let's add two more components to create a basic information extraction pipeline.

First, we will add the `clinlp_rule_based_entity_matcher`, along with some sample concepts to match:

```python
from clinlp.ie import Term

terms = {
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

entity_matcher = nlp.add_pipe(
    "clinlp_rule_based_entity_matcher", 
    config={"attr": "NORM", "fuzzy": 1}
)

entity_matcher.add_terms_from_dict(terms)
```

The above code adds three concepts to be matched (`prematuriteit`, `hypotensie`, and `veneus_infarct`), along with synonyms to match. Additionally, it configures the entity matcher on how to perform the matching. We have here configured the entity matcher to match against the `NORM` attribute by default, which it finds in the `Token.norm_` property the `clinlp_normalizer` set earlier. The `fuzzy` parameter specifies how much the concept text and the real text can differ (based on the edit distance). Some settings are overruled at the `Term` level. For instance, the `proximity=1` parameter for `bd verlaagd` specifies that at most one token may skipped between the words `bd` and `verlaagd`.

If we now process a piece of text, we can see that the entity recognizer has recognized some entities:

```python
text = (
    "Preterme neonaat (<p3) opgenomen, bd enigszins verlaagd, "
    "familieanamnese vermeldt eveneens hypotensie bij moeder. "
    "Thans geen aanwijzingen voor veneus infarkt wat ook geen "
    "verklaring voor de partus prematurus is. Risico op VI "
    "blijft aanwezig."
)

doc = nlp(text)

for ent in doc.spans['ents']:
    print(ent.text, ent.label_)

> 'Preterme' 'prematuriteit'
> '<p3' 'prematuriteit'
> 'bd enigszins verlaagd' 'hypotensie'
> 'hypotensie' 'hypotensie'
> 'veneus infarkt' 'veneus_infarct'
> 'partus prematurus' 'prematuriteit'
> 'VI' 'veneus_infarct'

```

As you can see, the `doc.spans['ents']` property now contains seven `Span` objects, each with the matched text, along with the concept label.

Now, as a final step, let's add the `clinlp_context_algorithm` component to the pipeline, which implements the Context Algorithm. For each matched entity, it can detect qualifiers, such as `Presence`, `Temporality` and `Experiencer`, based on triggers like 'geen', 'uitgesloten', etc.

```python
nlp.add_pipe("clinlp_context_algorithm", config={"phrase_matcher_attr": "NORM"})
```

We again configure it to match on the `NORM` attribute, set by the `clinlp_normalizer`.

If we now process the same text, we can see that the Context Algorithm has added some additional information to the entities:

```python
doc = nlp(text)

for ent in doc.spans['ents']:
    print(ent.text, ent._.qualifiers)


> 'Preterme' set()
> '<p3' set()
> 'bd enigszins verlaagd' set()
> 'hypotensie' {'Experiencer.Family'}
> 'veneus infarkt' {'Presence.Absent'}
> 'partus prematurus' set()
> 'VI' {'Temporality.Future'}
```

In the above example, for readability all default qualifier values (`Presence.Present`, `Temporality.Current`, `Experiencer.Patient`) have been omitted. You can see that three out of seven entities have correctly been qualified, either as `Absent`, related to `Family`, or potentially occurring in the `Future`. Of course, your specific use case determines how the output of this pipeline will further be handled.

## Conclusion

In this guide, we have shown how to create a blank model, add components to it, and process a piece of text. It also shows how to configure individual components and organize them in a specific information extraction pipeline. Note that there are more components available than shown in this example, you can find them on the [Components](components) page. By now you understand the basics, and are ready to further explore everything `clinlp` can offer!

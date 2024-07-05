# Metrics and statistics

`clinlp` contains calculators for some specific metrics and statistics for evaluating NLP tools. You can find some basic information on using them below.

## Information extraction

Information extraction related metrics and statistics for annotated datasets can be computed by using the `InfoExtractionDataset` and `InfoExtractionMetrics` classes. They require the following optional dependencies:

```bash
pip install clinlp[metrics]
```

### Creating a `InfoExtractionDataset`

An `InfoExtractionDataset` contains a collection of annotated documents, regardless of whether the annotations were collected manually, or from an NLP tool.

#### From `clinlp`

```python
from clinlp.metrics import InfoExtractionDataset
import clinlp
import spacy

# assumes a model (nlp) and iterable of texts (texts) exists
nlp_docs = nlp.pipe(texts)

clinlp_dataset = InfoExtractionDataset.from_clinlp_docs(nlp_docs)
```

#### From `MedCATTrainer`

The `MedCATTrainer` interface allows exporting annotated data in a `JSON` format. It can be converted to a `InfoExtractionDataset` as follows:

```python
from clinlp.metrics import InfoExtractionDataset
import json
from pathlib import Path

with Path('medcattrainer_export.json').open('rb') as f:
    mtrainer_data = json.load(f)

mct_dataset = InfoExtractionDataset.from_medcattrainer(mctrainer_data)
```

#### From `dict`

```python
from clinlp.metrics import InfoExtractionDataset

data = {
    "documents": [
        {
            "identifier": "...",
            "text": "...",
            "annotations": 
            {
                "text": "...",
                "start": ..., 
                "end": ...,
                "label": "...",
                "qualifiers": {
                    "...": "...",
                    ...
                }
            }, ...
        },
        ...
    ]
}

dict_dataset = InfoExtractionDataset.from_dict(data)
```

#### From `json`

```python
from clinlp.metrics import InfoExtractionDataset

json_dataset = InfoExtractionDataset.read_json("dataset.json")
```

Note that this method assumes a JSON file has been written by `InfoExtractionDataset.write_json`. We use a simple custom `json` format with all the information present, but please inform us if you know a more open format or standard to use here.

#### From other

If your data is in a different format, you can manually convert it by creating `Annotation` and `Document` objects, and add those to a `InfoExtractionDataset`. Below are some pointers on how to create the appropriate objects:

```python
from clinlp.metrics import Annotation, Document, InfoExtractionDataset

annotation = Annotation(
    text='prematuriteit',
    start=0,
    end=12,
    label='C0151526_prematuriteit',
    qualifiers={
        "Presence": "Present",
        "Temporality": "Current",
        "Experiencer": "Patient"
    }
)

document = Document(
    identifier='doc_0001',
    text='De patiënt heeft prematuriteit.',
    annotations=[annotation1, annotation2, ...]
)

dataset = InfoExtractionDataset(
    documents=[document1, document2, ...]
)

```

If you are writing code to convert data from a specific existing format, please consider contributing to `clinlp` by adding a `InfoExtractionDataset` method like `from_medcattrainer` and `from_clinlp_docs` that does the conversion.

#### Displaying descriptive statistics

Get descriptive statistics for an `InfoExtractionDataset` as follows:

```python
dataset.stats()

> {
    "num_docs": 50,
    "num_annotations": 513,
    "span_counts": {
        "prematuriteit": 43,
        "infectie": 31,
        "fototherapie": 25,
        "dysmaturiteit": 24,
        "IRDS": 20,
        "prematuur": 15,
        "sepsis": 15,
        "hyperbilirubinemie": 14,
        "Prematuriteit": 14,
        "ROP": 13,
        "necrotiserende enterocolitis": 12,
        "Prematuur": 11,
        "infektie": 11,
        "ductus": 11,
        "bloeding": 8,
        "dysmatuur": 7,
        "IUGR": 7,
        "Hyperbilirubinemie": 7,
        "transfusie": 6,
        "hyperbilirubinaemie": 6,
        "Dopamine": 6,
        "wisseltransfusie": 5,
        "premature partus": 5,
        "retinopathy of prematurity": 5,
        "bloedtransfusie": 5,
    },
    "label_counts": {
        "C0151526_prematuriteit": 94,
        "C0020433_hyperbilirubinemie": 68,
        "C0243026_sepsis": 63,
        "C0015934_intrauterine_groeivertraging": 57,
        "C0002871_anemie": 37,
        "C0035220_infant_respiratory_distress_syndrome": 25,
        "C0035344_retinopathie_van_de_prematuriteit": 21,
        "C0520459_necrotiserende_enterocolitis": 18,
        "C0013274_patent_ductus_arteriosus": 18,
        "C0020649_hypotensie": 18,
        "C0559477_perinatale_asfyxie": 18,
        "C0270191_intraventriculaire_bloeding": 17,
        "C0877064_post_hemorrhagische_ventrikeldilatatie": 13,
        "C0014850_oesophagus_atresie": 12,
        "C0006287_bronchopulmonale_dysplasie": 9,
        "C0031190_persisterende_pulmonale_hypertensie": 7,
        "C0015938_macrosomie": 6,
        "C0751954_veneus_infarct": 5,
        "C0025289_meningitis": 5,
        "C0023529_periventriculaire_leucomalacie": 2,
    },
    "qualifier_counts": {
        "Presence": {"Present": 436, "Uncertain": 34, "Absent": 30},
        "Temporality": {"Current": 473, "Historical": 18, "Future": 9},
        "Experiencer": {"Patient": 489, "Family": 9, "Other": 2},
    }
}
```

You can also get the individual statistics, rather than all combined in a dictionary, i.e.:

```python
dataset.num_docs()

> 50
```

### Comparison statistics

To compare two `InfoExtractionDataset` objects, you need to create a `InfoExtractionMetrics` object that compares two datasets. The `InfoExtractionMetrics` object will then calculate the relevant metrics for the annotations the two datasets.

```python
from clinlp.metrics import InfoExtractionMetrics

nlp_metrics = InfoExtractionMetrics(dataset1, dataset2)
```

#### Entity metrics

For comparison metrics on entities, use:

```python
nlp_metrics.entity_metrics()

> {
    'ent_type': {
        'correct': 480,
        'incorrect': 1,
        'partial': 0,
        'missed': 32,
        'spurious': 21,
        'possible': 513,
        'actual': 502,
        'precision': 0.9561752988047809,
        'recall': 0.935672514619883,
        'f1': 0.9458128078817734
    },
    'partial': {
        'correct': 473,
        'incorrect': 0,
        'partial': 8,
        'missed': 32,
        'spurious': 21,
        'possible': 513,
        'actual': 502,
        'precision': 0.950199203187251,
        'recall': 0.9298245614035088,
        'f1': 0.9399014778325123
    },
    'strict': {
        'correct': 473,
        'incorrect': 8,
        'partial': 0,
        'missed': 32,
        'spurious': 21,
        'possible': 513,
        'actual': 502,
        'precision': 0.9422310756972112,
        'recall': 0.9220272904483431,
        'f1': 0.9320197044334976
    },
    'exact': {
        'correct': 473,
        'incorrect': 8,
        'partial': 0,
        'missed': 32,
        'spurious': 21,
        'possible': 513,
        'actual': 502,
        'precision': 0.9422310756972112,
        'recall': 0.9220272904483431,
        'f1': 0.9320197044334976
    }
}
```

The different metrics (`partial`, `exact`, `strict` and `ent_type`) are calculated using `Nervaluate`, based on the SemEval 2013 - 9.1 task. Check the  [Nervaluate documentation](https://github.com/MantisAI/nervaluate) for more information.

#### Qualifier metrics

For comparison metrics on qualifiers, use:

```python
nlp_metrics.qualifier_info()

> {
    "Experiencer": {
        "metrics": {
            "n": 460,
            "precision": 0.3333333333333333,
            "recall": 0.09090909090909091,
            "f1": 0.14285714285714288,
        },
        "misses": [
            {
                "doc.identifier": "doc_0001",
                "annotation": {
                    "text": "anemie",
                    "start": 1849,
                    "end": 1855,
                    "label": "C0002871_anemie",
                },
                "true_qualifier": "Family",
                "pred_qualifier": "Patient",
            },
            ...,
        ],
    },
    "Temporality": {
        "metrics": {"n": 460, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "misses": [
            {
                "doc.identifier": "doc_0001",
                "annotation": {
                    "text": "premature partus",
                    "start": 1611,
                    "end": 1627,
                    "label": "C0151526_prematuriteit",
                },
                "true_qualifier": "Current",
                "pred_qualifier": "Historical",
            },
            ...,
        ],
    },
    "Plausibility": {
        "metrics": {
            "n": 460,
            "precision": 0.6486486486486487,
            "recall": 0.5217391304347826,
            "f1": 0.5783132530120482,
        },
        "misses": [
            {
                "doc.identifier": "doc_0001",
                "annotation": {
                    "text": "Groeivertraging",
                    "start": 1668,
                    "end": 1683,
                    "label": "C0015934_intrauterine_groeivertraging",
                },
                "true_qualifier": "Current",
                "pred_qualifier": "Future",
            },
            ...,
        ],
    },
    "Negation": {
        "metrics": {
            "n": 460,
            "precision": 0.7692307692307693,
            "recall": 0.6122448979591837,
            "f1": 0.6818181818181818,
        },
        "misses": [
            {
                "doc.identifier": "doc_0001",
                "annotation": {
                    "text": "wisseltransfusie",
                    "start": 4095,
                    "end": 4111,
                    "label": "C0020433_hyperbilirubinemie",
                },
                "true_qualifier": "Present",
                "pred_qualifier": "Absent",
            },
            ...,
        ]
    }
}
```

# Data

The `clinlp` repository contains some open data files with real (or semi-real) examples. Some of these are used by `clinlp` (for example in the tests), but they are also available for others to use.

The files are located at the [Data directory in the GitHub repo](https://github.com/umcu/clinlp/tree/main/data).

## `tokenizer_cases.json`

Some cases for testing tokenizers, collected during development of clinlp, often based on real examples.

## `sentencizer_cases.json`

Some cases for testing sentencizers, collected during development of clinlp, often based on real examples.

## `qualifier_cases.json`

Some cases for testing qualifier detectors, collected during development of clinlp, often based on real examples. Each doc contains exactly one entity, which makes it easier for our regression tests to mark skips.

You can load the dataset to an `InfoExtractionDataset` for further evaluation using:

```python
from clinlp.data import InfoExtractionDataset

dataset = InfoExtractionDataset.from_json("data/qualifier_cases.json")
```


## `mantra_gsc.json`

A re-release of the Dutch part of the Mantra GSC corpus, which can be used for evaluating algorithms that match and link to UMLS. For more information, see: [Kors et al., 2015](https://doi.org/10.1093/jamia/ocv037).

You can load the dataset to an `InfoExtractionDataset` for further evaluation using:

```python
from clinlp.data import InfoExtractionDataset

dataset = InfoExtractionDataset.from_json("data/mantra_gsc.json")
```

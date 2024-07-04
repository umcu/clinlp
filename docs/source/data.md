# Data

The `clinlp` repository contains some open data files with real (or semi-real) examples. Some of these are used by `clinlp` (for example in the tests), but they are also available for others to use.

The files are located at the [Data directory in the GitHub repo](https://github.com/umcu/clinlp/tree/main/data).

## `tokenizer_cases.json`

Some cases for testing tokenizers, collected during development of clinlp, often based on real examples.

## `sentencizer_cases.json`

Some cases for testing sentencizers, collected during development of clinlp, often based on real examples.

## `qualifier_cases.json`

Some cases for testing qualifier detectors, collected during development of clinlp, often based on real examples. Each doc contains exactly one entity, which makes it easier for our regression tests to mark skips.

You can load this file to an `InfoExtractionDataset` for further evaluation using: 

```python
from clinlp.data import InfoExtractionDataset

dataset = InfoExtractionDataset.from_json("data/qualifier_cases.json")
```

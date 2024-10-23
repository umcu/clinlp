import json
from collections.abc import Collection
from pathlib import Path

import pytest

from clinlp.metrics import InfoExtractionDataset


def load_examples(filename: str) -> list[dict]:
    with Path(filename).open("rb") as file:
        return json.load(file)["examples"]


def load_qualifier_examples(
    filename: str, failures=Collection[int]
) -> list["pytest.param"]:
    ied = InfoExtractionDataset.read_json(filename)

    examples_as_param = []

    for doc in ied.docs:
        marks = pytest.mark.xfail if doc.identifier in failures else []

        examples_as_param.append(
            pytest.param(
                doc.text,
                doc.annotations[0],
                id=f"qualifier_case_{doc.identifier}",
                marks=marks,
            )
        )

    return examples_as_param

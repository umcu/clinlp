import json
from pathlib import Path
from typing import Collection

import pytest

from tests.conftest import TEST_DATA_DIR


def load_examples(filename: str) -> list[dict]:
    with Path.open(TEST_DATA_DIR / filename, "rb") as file:
        return json.load(file)["examples"]


def load_qualifier_examples(
    filename: str, failures=Collection[int]
) -> list["pytest.param"]:
    examples = load_examples(filename)

    examples_as_param = []

    for example in examples:
        marks = pytest.mark.xfail if example["example_id"] in failures else []

        examples_as_param.append(
            pytest.param(
                example["text"],
                example["ent"],
                id=f"qualifier_case_{example['example_id']}",
                marks=marks,
            )
        )

    return examples_as_param

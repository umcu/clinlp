import json

from clinlp.sentencizer import ClinlpSentencizer


class MockToken:
    def __init__(self, text: str):
        self.text = text
        self.is_sent_start = False


def get_mock_tokens(texts: list[str]):
    return [MockToken(text) for text in texts]


class TestClinlpSentencizerRegression:
    def test_default_clinlp_sentencizer_examples(self):
        sentencizer = ClinlpSentencizer()

        with open("tests/data/sentencizer_cases.json", "rb") as file:
            data = json.load(file)["data"]

        for example in data:
            tokens = get_mock_tokens(example["tokens"])
            assert sentencizer.predict(tokens) == example["sentence_starts"]
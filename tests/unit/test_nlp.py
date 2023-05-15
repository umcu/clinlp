from clinlp.clinlp import create_model


class TestUnitCreateModel:
    def test_create_model(self):
        _ = create_model()

    def test_apply_model(self):
        nlp = create_model()
        nlp("dit is een test")

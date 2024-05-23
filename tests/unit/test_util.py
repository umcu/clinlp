import pytest
import spacy
from spacy import Language
from thinc.api import ConfigValidationError

import clinlp  # noqa: F401
from clinlp.util import (
    clinlp_autocomponent,
    get_class_init_signature,
    interval_dist,
)


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


def add_pipe_for_test(*args, **kwargs):
    nlp = spacy.blank("clinlp")
    component = nlp.add_pipe(*args, **kwargs)
    return component


class TestUnitGetClassInitSignature:
    def test_args_only(self):
        # Arrange
        class MyClass:
            def __init__(self, a, b, c):
                pass

        # Act
        args, kwargs = get_class_init_signature(MyClass)

        # Assert
        assert args == ["a", "b", "c"]
        assert kwargs == {}

    def test_kwargs_only(self):
        # Arrange
        class MyClass:
            def __init__(self, a=1, b=2, c=3):
                pass

        # Act
        args, kwargs = get_class_init_signature(MyClass)

        # Assert
        assert args == []
        assert kwargs == {"a": 1, "b": 2, "c": 3}

    def test_mixed(self):
        # Arrange
        class MyClass:
            def __init__(self, a, b="test"):
                pass

        # Act
        args, kwargs = get_class_init_signature(MyClass)

        # Assert
        assert args == ["a"]
        assert kwargs == {"b": "test"}


class TestUnitClinlpAutocomponent:
    def test_only_args(self, nlp):
        # Arrange
        @Language.factory(name="myclass")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, nlp, name):
                self.nlp = nlp
                self.name = name

        # Act & Assert
        assert MyClass(nlp="nlp", name="name").name == "name"
        assert nlp.add_pipe("myclass").name == "myclass"
        assert nlp.add_pipe("myclass", name="name").name == "name"

        with pytest.raises(TypeError):
            MyClass()

        with pytest.raises(TypeError):
            MyClass(name="bla")

        with pytest.raises(TypeError):
            MyClass("bla")

    def test_only_kwargs(self, nlp):
        # Arrange
        @Language.factory(name="myclass_test1")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, setting_1=32, setting_2="max"):
                self.setting_1 = setting_1
                self.setting_2 = setting_2

        # Act & Assert
        assert MyClass(nlp="nlp", name="test").setting_1 == 32
        assert MyClass(nlp="nlp", name="test").setting_2 == "max"
        assert MyClass(setting_1=64, setting_2="min").setting_1 == 64
        assert MyClass(setting_1=64, setting_2="min").setting_2 == "min"
        assert (
            MyClass(nlp="nlp", name="test", setting_1=64, setting_2="min").setting_1
            == 64
        )
        assert (
            MyClass(nlp="nlp", name="test", setting_1=64, setting_2="min").setting_2
            == "min"
        )

        assert add_pipe_for_test("myclass_test1", name="test").setting_1 == 32
        assert add_pipe_for_test("myclass_test1", name="test").setting_2 == "max"
        assert (
            add_pipe_for_test("myclass_test1", config={"setting_1": 64}).setting_1 == 64
        )
        assert (
            add_pipe_for_test("myclass_test1", config={"setting_2": "min"}).setting_2
            == "min"
        )
        assert (
            add_pipe_for_test(
                "myclass_test1", config={"setting_1": 64}, name="test"
            ).setting_1
            == 64
        )
        assert (
            add_pipe_for_test(
                "myclass_test1", config={"setting_2": "min"}, name="test"
            ).setting_2
            == "min"
        )

        with pytest.raises(TypeError):
            MyClass(setting_3="None")

        with pytest.raises(ConfigValidationError):
            add_pipe_for_test("myclass_test1", config={"setting_3": "None"})

    def test_mixed_args_and_kwargs(self):
        # Arrange
        @Language.factory(name="myclass_test2")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, name, setting_1=32):
                self.name = name
                self.setting_1 = setting_1

        # Act & Assert
        assert MyClass(nlp="nlp", name="test").setting_1 == 32
        assert MyClass(nlp="nlp", name="test").name == "test"
        assert MyClass(name="test", setting_1=64).setting_1 == 64
        assert MyClass(name="test", setting_1=64).name == "test"
        assert add_pipe_for_test("myclass_test2", name="test").setting_1 == 32
        assert add_pipe_for_test("myclass_test2", name="test").name == "test"
        assert (
            add_pipe_for_test("myclass_test2", config={"setting_1": 64}).setting_1 == 64
        )
        assert (
            add_pipe_for_test(
                "myclass_test2", config={"setting_1": 64}, name="test"
            ).setting_1
            == 64
        )
        assert (
            add_pipe_for_test(
                "myclass_test2", config={"setting_1": 64}, name="test"
            ).name
            == "test"
        )

        with pytest.raises(TypeError):
            MyClass(setting_1=10)

        with pytest.raises(ConfigValidationError):
            add_pipe_for_test("myclass_test2", config={"setting_3": "None"})

    def test_default_args(self):
        # Arrange
        _defaults = {"setting_1": 1024}

        @Language.factory(name="myclass_test3", default_config=_defaults)
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, setting_1=_defaults["setting_1"]):
                self.setting_1 = setting_1

        # Act & Assert
        assert MyClass().setting_1 == 1024
        assert add_pipe_for_test("myclass_test3").setting_1 == 1024
        assert (
            add_pipe_for_test("myclass_test3", config={"setting_1": 2048}).setting_1
            == 2048
        )

    def test_with_inheritance(self):
        # Arrange
        _base_defaults = {"base_arg": 1}
        _sub_defaults = {"sub_arg": 2}

        class MyClassBase:
            def __init__(self, base_arg=_base_defaults["base_arg"]):
                self.base_arg = base_arg

        @Language.factory(
            name="myclass_test4", default_config=_sub_defaults | _base_defaults
        )
        @clinlp_autocomponent
        class MyClass(MyClassBase):
            def __init__(self, sub_arg=_sub_defaults["sub_arg"], **kwargs):
                self.sub_arg = sub_arg
                super().__init__(**kwargs)

        # Act & Assert
        assert MyClass().sub_arg == 2
        assert MyClass().base_arg == 1
        assert MyClass(sub_arg=20).sub_arg == 20
        assert MyClass(base_arg=10).base_arg == 10
        assert MyClass(sub_arg=20, base_arg=10).sub_arg == 20
        assert MyClass(sub_arg=20, base_arg=10).base_arg == 10
        assert add_pipe_for_test("myclass_test4").sub_arg == 2
        assert add_pipe_for_test("myclass_test4").base_arg == 1
        assert add_pipe_for_test("myclass_test4", config={"sub_arg": 20}).sub_arg == 20
        assert (
            add_pipe_for_test("myclass_test4", config={"base_arg": 10}).base_arg == 10
        )
        assert (
            add_pipe_for_test(
                "myclass_test4", config={"sub_arg": 20, "base_arg": 10}
            ).sub_arg
            == 20
        )
        assert (
            add_pipe_for_test(
                "myclass_test4", config={"sub_arg": 20, "base_arg": 10}
            ).base_arg
            == 10
        )


class TestUnitIntervalDistance:
    @pytest.mark.parametrize(
        "a, b, c, d, expected_dist",
        [
            (0, 10, 12, 20, 2),
            (0, 10, 10, 20, 0),
            (12, 20, 0, 10, 2),
            (10, 20, 0, 10, 0),
            (0, 10, 5, 15, 0),
        ],
    )
    def test_interval_distance(self, a, b, c, d, expected_dist):
        # Arrange, Act
        dist = interval_dist(a, b, c, d)

        # Assert
        assert dist == expected_dist

    def test_interval_distance_unhappy(self):
        # Arrange, Act & Assert
        with pytest.raises(ValueError):
            _ = interval_dist(5, 0, 5, 0)

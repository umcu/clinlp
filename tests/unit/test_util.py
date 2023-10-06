import pytest
import spacy
from spacy import Language
from thinc.api import ConfigValidationError

import clinlp  # noqa: F401
from clinlp.util import _UnusedArgument, clinlp_autocomponent, get_class_init_signature


@pytest.fixture
def nlp():
    return spacy.blank("clinlp")


def add_pipe_for_test(*args, **kwargs):
    nlp = spacy.blank("clinlp")
    component = nlp.add_pipe(*args, **kwargs)
    return component


class TestUnitUnusedArgument:
    def test_unused_argument(self):
        assert _UnusedArgument


class TestUnitGetClassInitSignature:
    def test_args_only(self):
        class MyClass:
            def __init__(self, a, b, c):
                pass

        args, kwargs = get_class_init_signature(MyClass)

        assert args == ["a", "b", "c"]
        assert kwargs == {}

    def test_kwargs_only(self):
        class MyClass:
            def __init__(self, a=1, b=2, c=3):
                pass

        args, kwargs = get_class_init_signature(MyClass)

        assert args == []
        assert kwargs == {"a": 1, "b": 2, "c": 3}

    def test_mixed(self):
        class MyClass:
            def __init__(self, a, b="test"):
                pass

        args, kwargs = get_class_init_signature(MyClass)

        assert args == ["a"]
        assert kwargs == {"b": "test"}


class TestUnitClinlpAutocomponent:
    def test_only_args(self, nlp):
        @Language.factory(name="myclass")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, nlp, name):
                self.nlp = nlp
                self.name = name

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
        @Language.factory(name="myclass_test1")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, setting_1=32, setting_2="max"):
                self.setting_1 = setting_1
                self.setting_2 = setting_2

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
        @Language.factory(name="myclass_test2")
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, name, setting_1=32):
                self.name = name
                self.setting_1 = setting_1

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
        _defaults = {"setting_1": 1024}

        @Language.factory(name="myclass_test3", default_config=_defaults)
        @clinlp_autocomponent
        class MyClass:
            def __init__(self, setting_1=_defaults["setting_1"]):
                self.setting_1 = setting_1

        assert MyClass().setting_1 == 1024
        assert add_pipe_for_test("myclass_test3").setting_1 == 1024
        assert (
            add_pipe_for_test("myclass_test3", config={"setting_1": 2048}).setting_1
            == 2048
        )

    def test_with_inheritance(self):
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

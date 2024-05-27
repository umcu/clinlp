import pytest
from spacy import Language
from thinc.api import ConfigValidationError

import clinlp  # noqa: F401
from clinlp.util import (
    clinlp_autocomponent,
    get_class_init_signature,
    interval_dist,
)


# Arrange
@pytest.fixture(scope="session")
def component_1():
    @Language.factory(name="test_component_1")
    @clinlp_autocomponent
    class TestComponent1:
        def __init__(self, nlp, name):
            self.nlp = nlp
            self.name = name

    return TestComponent1


# Arrange
@pytest.fixture(scope="session")
def component_2():
    @Language.factory(name="test_component_2")
    @clinlp_autocomponent
    class TestComponent2:
        def __init__(self, setting_1=32, setting_2="max"):
            self.setting_1 = setting_1
            self.setting_2 = setting_2

    return TestComponent2


# Arrange
@pytest.fixture(scope="session")
def component_3():
    @Language.factory(name="test_component_3")
    @clinlp_autocomponent
    class TestComponent3:
        def __init__(self, name, setting_1=32):
            self.name = name
            self.setting_1 = setting_1

    return TestComponent3


# Arrange
@pytest.fixture(scope="session")
def component_4():
    _defaults = {"setting_1": 1024}

    @Language.factory(name="test_component_4", default_config=_defaults)
    @clinlp_autocomponent
    class TestComponent4:
        def __init__(self, setting_1=_defaults["setting_1"]):
            self.setting_1 = setting_1

    return TestComponent4


# Arrange
@pytest.fixture(scope="session")
def component_5():
    _base_defaults = {"base_arg": 1}
    _sub_defaults = {"sub_arg": 2}

    class TestComponent5Base:
        def __init__(self, base_arg=_base_defaults["base_arg"]):
            self.base_arg = base_arg

    @Language.factory(
        name="test_component_5", default_config=_sub_defaults | _base_defaults
    )
    @clinlp_autocomponent
    class TestComponent5(TestComponent5Base):
        def __init__(self, sub_arg=_sub_defaults["sub_arg"], **kwargs):
            self.sub_arg = sub_arg
            super().__init__(**kwargs)

    return TestComponent5


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
    def test_only_args_class(self, component_1):
        # Act
        component = component_1(nlp="nlp", name="name")

        # Assert
        assert component.name == "name"

    @pytest.mark.parametrize(
        "kwargs, attr, expected_value",
        [
            ({}, "name", "test_component_1"),
            ({"name": "name"}, "name", "name"),
        ],
    )
    def test_only_args_pipe(self, nlp, component_1, kwargs, attr, expected_value):
        # Act
        component = nlp.add_pipe("test_component_1", **kwargs)

        # Assert
        assert getattr(component, attr) == expected_value

    def test_only_args_pipe_error_1(self, component_1):
        # Assert
        with pytest.raises(TypeError):
            # Act
            component_1()

    def test_only_args_pipe_error_2(self, component_1):
        # Assert
        with pytest.raises(TypeError):
            # Act
            component_1(name="bla")

    def test_only_args_pipe_error_3(self, component_1):
        # Assert
        with pytest.raises(TypeError):
            # Act
            component_1("bla")

    @pytest.mark.parametrize(
        "kwargs, attr, expected_value",
        [
            ({"nlp": "nlp", "name": "test"}, "setting_1", 32),
            ({"nlp": "nlp", "name": "test"}, "setting_2", "max"),
            ({"setting_1": 64, "setting_2": "min"}, "setting_1", 64),
            ({"setting_1": 64, "setting_2": "min"}, "setting_2", "min"),
            (
                {"nlp": "nlp", "name": "test", "setting_1": 64, "setting_2": "min"},
                "setting_1",
                64,
            ),
            (
                {"nlp": "nlp", "name": "test", "setting_1": 64, "setting_2": "min"},
                "setting_2",
                "min",
            ),
        ],
    )
    def test_only_kwargs_class(self, component_2, kwargs, attr, expected_value):
        # Act
        component = component_2(**kwargs)

        # Assert
        assert getattr(component, attr) == expected_value

    @pytest.mark.parametrize(
        "kwargs, config, attr, expected_value",
        [
            ({"name": "test"}, {}, "setting_1", 32),
            ({"name": "test"}, {}, "setting_2", "max"),
            ({}, {"setting_1": 64}, "setting_1", 64),
            ({}, {"setting_2": "min"}, "setting_2", "min"),
            ({"name": "test"}, {"setting_1": 64}, "setting_1", 64),
            ({"name": "test"}, {"setting_2": "min"}, "setting_2", "min"),
        ],
    )
    def test_only_kwargs_pipe(
        self, nlp, component_2, kwargs, config, attr, expected_value
    ):
        # Act
        component = nlp.add_pipe("test_component_2", **kwargs, config=config)

        # Assert
        assert getattr(component, attr) == expected_value

    def test_only_kwargs_pipe_error_1(self, component_2):
        # Assert
        with pytest.raises(TypeError):
            # Act
            component_2(setting_3="None")

    def test_only_kwargs_pipe_error_2(self, nlp, component_2):
        # Assert
        with pytest.raises(ConfigValidationError):
            # Act
            nlp.add_pipe("test_component_2", config={"setting_3": "None"})

    @pytest.mark.parametrize(
        "kwargs, attr, expected_value",
        [
            ({"nlp": "nlp", "name": "test"}, "setting_1", 32),
            ({"nlp": "nlp", "name": "test"}, "name", "test"),
            ({"name": "test", "setting_1": 64}, "setting_1", 64),
            ({"name": "test", "setting_1": 64}, "name", "test"),
        ],
    )
    def test_mixed_args_and_kwargs_class(
        self, component_3, kwargs, attr, expected_value
    ):
        # Act
        component = component_3(**kwargs)

        # Assert
        assert getattr(component, attr) == expected_value

    @pytest.mark.parametrize(
        "kwargs, config, attr, expected_value",
        [
            ({"name": "test"}, {}, "setting_1", 32),
            ({"name": "test"}, {}, "name", "test"),
            ({}, {"setting_1": 64}, "setting_1", 64),
            ({"name": "test"}, {"setting_1": 64}, "setting_1", 64),
            ({"name": "test"}, {"setting_1": 64}, "name", "test"),
        ],
    )
    def test_mixed_args_and_kwargs_pipe(
        self, nlp, component_3, kwargs, config, attr, expected_value
    ):
        # Act
        component = nlp.add_pipe("test_component_3", **kwargs, config=config)

        # Assert
        assert getattr(component, attr) == expected_value

    def test_mixed_args_and_kwargs_pipe_error_1(self, component_3):
        # Assert
        with pytest.raises(TypeError):
            # Act
            component_3(setting_1=10)

    def test_mixed_args_and_kwargs_pipe_error_2(self, nlp, component_3):
        # Assert
        with pytest.raises(ConfigValidationError):
            # Act
            nlp.add_pipe("test_component_3", config={"setting_3": "None"})

    def test_default_args_component(self, component_4):
        # Act
        component = component_4()

        # Assert
        assert component.setting_1 == 1024

    @pytest.mark.parametrize(
        "config, attr, expected_value",
        [
            ({}, "setting_1", 1024),
            ({"setting_1": 2048}, "setting_1", 2048),
        ],
    )
    def test_default_args_pipe(self, nlp, component_4, config, attr, expected_value):
        # Act
        component = nlp.add_pipe("test_component_4", config=config)

        # Assert
        assert getattr(component, attr) == expected_value

    @pytest.mark.parametrize(
        "kwargs, attr, expected_value",
        [
            ({}, "sub_arg", 2),
            ({}, "base_arg", 1),
            ({"sub_arg": 20}, "sub_arg", 20),
            ({"base_arg": 10}, "base_arg", 10),
            ({"sub_arg": 20, "base_arg": 10}, "sub_arg", 20),
            ({"sub_arg": 20, "base_arg": 10}, "base_arg", 10),
        ],
    )
    def test_with_inheritance_class(self, component_5, kwargs, attr, expected_value):
        # Act
        component = component_5(**kwargs)

        # Assert
        assert getattr(component, attr) == expected_value

    @pytest.mark.parametrize(
        "config, attr, expected_value",
        [
            ({}, "sub_arg", 2),
            ({}, "base_arg", 1),
            ({"sub_arg": 20}, "sub_arg", 20),
            ({"base_arg": 10}, "base_arg", 10),
            ({"sub_arg": 20, "base_arg": 10}, "sub_arg", 20),
            ({"sub_arg": 20, "base_arg": 10}, "base_arg", 10),
        ],
    )
    def test_with_inheritance_pipe(
        self, nlp, component_5, config, attr, expected_value
    ):
        # Act
        component = nlp.add_pipe("test_component_5", config=config)

        # Assert
        assert getattr(component, attr) == expected_value


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
        # Act
        dist = interval_dist(a, b, c, d)

        # Assert
        assert dist == expected_dist

    def test_interval_distance_error(self):
        # Assert
        with pytest.raises(ValueError):
            # Act
            _ = interval_dist(10, 0, 0, 10)

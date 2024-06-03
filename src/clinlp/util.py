"""Utility functions that are used across the library."""

import inspect
from inspect import Parameter, Signature
from typing import Callable, Tuple, Type

from makefun import with_signature
from spacy.language import Language


class _UnusedArgument:
    """Placeholder for unused arguments in the signature of a function."""


def get_class_init_signature(cls: Type) -> Tuple[list, dict]:
    """
    Get the arguments and keyword arguments of a class's __init__ method.

    Parameters
    ----------
    cls
        The class to get the signature of.

    Returns
    -------
        The arguments and keyword arguments of the class's __init__ method.
    """
    args = []
    kwargs = {}

    for mro_class in cls.__mro__:
        if "__init__" in mro_class.__dict__:
            argspec = inspect.getfullargspec(mro_class)

            if argspec.defaults is None:
                args += argspec.args[1:]
            else:
                first_arg_wit_default = len(argspec.args) - len(argspec.defaults)
                args += argspec.args[1:first_arg_wit_default]
                kwargs |= dict(
                    zip(argspec.args[first_arg_wit_default:], argspec.defaults)
                )

    return args, kwargs


def clinlp_component(*args, register: bool = True, **kwargs) -> Callable:
    """
    Denote a class as clinlp component.

    Should be used as a decorator on a class. Additionally handles the `name` and `nlp`
    arguments, and handles inheritance.

    Parameters
    ----------
    register, optional
        Whether to automatically register the class as a spaCy component, or to only
        return the make function, by default `True`. If set to `False`, should
        probably be further decorated with `@Language.factory`.

    Returns
    -------
        The make function for the class.
    """

    def _clinlp_component(cls: Type) -> Callable:
        component_args, component_kwargs = get_class_init_signature(cls)

        make_component_args = component_args.copy()

        if "nlp" not in make_component_args:
            make_component_args = ["nlp", *make_component_args]

        if "name" not in make_component_args:
            make_component_args = ["name", *make_component_args]

        params = [
            Parameter(
                arg, kind=Parameter.POSITIONAL_OR_KEYWORD, default=_UnusedArgument()
            )
            for arg in make_component_args
        ]

        for kwarg, default in component_kwargs.items():
            params.append(
                Parameter(kwarg, kind=Parameter.POSITIONAL_OR_KEYWORD, default=default)
            )

        @with_signature(Signature(params), func_name="make_component")
        def make_component(*args, **kwargs) -> Type:
            if len(args) > 0:
                msg = "Please pass all arguments as keywords."
                raise RuntimeError(msg)

            cls_kwargs = {
                k: v
                for k, v in kwargs.items()
                if (k in component_args or k in component_kwargs)
                and (not isinstance(v, _UnusedArgument))
            }

            return cls(**cls_kwargs)

        if register:
            Language.factory(*args, func=make_component, **kwargs)

        return make_component

    return _clinlp_component


def interval_dist(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    """
    Calculate the distance between two intervals.

    Parameters
    ----------
    start_a
        The start of the first interval.
    end_a
        The end of the first interval.
    start_b
        The start of the second interval.
    end_b
        The end of the second interval.

    Returns
    -------
        The distance between the two intervals.

    Raises
    ------
    ValueError
        If an input interval is malformed.
    """
    if (end_a < start_a) or (end_b < start_b):
        msg = "Input malformed interval."
        raise ValueError(msg)

    return max(0, start_a - end_b, start_b - end_a)

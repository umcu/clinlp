"""Utility functions that are used throughout the library."""

import inspect
from collections.abc import Callable
from inspect import Parameter, Signature

from makefun import with_signature
from spacy.language import Language


class _UnusedArgument:
    """Placeholder for unused arguments in the signature of a function."""


def get_class_init_signature(cls: type) -> tuple[list, dict]:
    """
    Get the arguments and defaults of a class's ``__init__`` method.

    Handles inheritance.

    Parameters
    ----------
    cls
        The class to get the signature of.

    Returns
    -------
    ``list``
        The arguments of the class's ``__init__`` method.
    ``dict``
        A mapping of arguments to their defaults, for those arguments that have one.
    """
    args = []
    defaults = {}

    for mro_class in cls.__mro__:
        if "__init__" in mro_class.__dict__:
            argspec = inspect.getfullargspec(mro_class)
            init_defaults = argspec.defaults or []

            args += argspec.args[1:]
            defaults |= dict(
                zip(
                    argspec.args[len(argspec.args) - len(init_defaults) :],
                    init_defaults,
                    strict=False,
                )
            )

            if argspec.kwonlyargs is not None:
                args += argspec.kwonlyargs

            if argspec.kwonlydefaults is not None:
                defaults |= argspec.kwonlydefaults

    return args, defaults


def clinlp_component(*args, **kwargs) -> Callable:
    """
    Register a ``clinlp`` component with ``spaCy``.

    Should be used as a decorator on a class. Additionally handles the ``name`` and
    ``nlp`` arguments, and handles inheritance.

    Returns
    -------
    ``Callable``
        The decorated class.
    """

    def _clinlp_component(cls: type) -> Callable[[list, dict], type]:
        component_args, component_defaults = get_class_init_signature(cls)

        make_component_args = component_args.copy()

        if "nlp" not in make_component_args:
            make_component_args = ["nlp", *make_component_args]

        if "name" not in make_component_args:
            make_component_args = ["name", *make_component_args]

        params = [
            Parameter(
                arg,
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=component_defaults.get(arg, _UnusedArgument()),
            )
            for arg in make_component_args
        ]

        @with_signature(Signature(params), func_name="make_component")
        def make_component(*args, **kwargs) -> type:
            if len(args) > 0:
                msg = "Please pass all arguments as keywords."
                raise RuntimeError(msg)

            cls_kwargs = {
                k: v
                for k, v in kwargs.items()
                if (k in component_args) and (not isinstance(v, _UnusedArgument))
            }

            return cls(**cls_kwargs)

        Language.factory(
            *args, func=make_component, default_config=component_defaults, **kwargs
        )

        return cls

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
    ``int``
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

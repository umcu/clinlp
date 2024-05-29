import inspect
from inspect import Parameter, Signature
from typing import Callable, Tuple, Type

from makefun import with_signature
from spacy.language import Language


class _UnusedArgument:
    pass


def get_class_init_signature(cls: Type) -> Tuple[list, dict]:
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
    if (end_a < start_a) or (end_b < start_b):
        msg = "Input malformed interval."
        raise ValueError(msg)

    return max(0, start_a - end_b, start_b - end_a)

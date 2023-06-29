import inspect
from inspect import Parameter, Signature

from makefun import with_signature


class _UnusedArgument:
    pass


def get_class_init_signature(cls):
    args = []
    kwargs = {}

    for C in cls.__mro__:
        if "__init__" in C.__dict__:
            argspec = inspect.getfullargspec(C)

            print(argspec)


            if argspec.defaults is None:
                args += argspec.args[1:]
            else:
                args += argspec.args[1 : len(argspec.defaults)-1]
                kwargs |= dict(zip(argspec.args[len(argspec.defaults)-1 :], argspec.defaults))

    return args, kwargs


def clinlp_autocomponent(cls):
    component_args, component_kwargs = get_class_init_signature(cls)

    params = []

    make_component_args = component_args.copy()

    if "nlp" not in make_component_args:
        make_component_args = ["nlp"] + make_component_args

    if "name" not in make_component_args:
        make_component_args = ["name"] + make_component_args

    for arg in make_component_args:
        params.append(Parameter(arg, kind=Parameter.POSITIONAL_OR_KEYWORD, default=_UnusedArgument()))

    for kwarg, default in component_kwargs.items():
        params.append(Parameter(kwarg, kind=Parameter.POSITIONAL_OR_KEYWORD, default=default))

    @with_signature(Signature(params), func_name="make_component")
    def make_component(*args, **kwargs):
        if len(args) > 0:
            raise RuntimeError("Please pass all arguments as keywords.")

        cls_kwargs = {
            k: v
            for k, v in kwargs.items()
            if (k in component_args or k in component_kwargs) and (not isinstance(v, _UnusedArgument))
        }

        print("cls_kwargs=", cls_kwargs)

        return cls(**cls_kwargs)

    return make_component

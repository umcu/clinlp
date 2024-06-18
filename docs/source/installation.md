# Installation

The easiest way to install `clinlp` is by using `pip`:

```bash
pip install clinlp
```

As a good practice, we recommend installing `clinlp` in a virtual environment. If you are not familiar with virtual environments, you can find more information [here](https://docs.python.org/3/library/venv.html).

## Optional dependencies

To keep the base package lightweight, we use optional dependencies for some components. In the component library, each component will list the required optional dependencies, if any. They can be installed using:

```bash
pip install clinlp[extra_name]
```

Or, if you want to install multiple extras at once:

```bash
pip install clinlp[extra_name1,extra_name2]
```

# Contributing to `clinlp`

- [Contributing to `clinlp`](#contributing-to-clinlp)
  - [Introduction](#introduction)
  - [Contact](#contact)
    - [Questions](#questions)
    - [Bug reports](#bug-reports)
    - [Feature requests and contributions](#feature-requests-and-contributions)
  - [Pull requests](#pull-requests)
  - [Roadmap](#roadmap)
  - [Developer environment setup](#developer-environment-setup)
    - [Repository structure](#repository-structure)
  - [Coding standards](#coding-standards)
    - [General principles](#general-principles)
    - [Creating a component](#creating-a-component)
    - [Formatting and linting](#formatting-and-linting)
    - [Dependencies](#dependencies)
    - [Tests](#tests)
      - [Arrange, act, assert](#arrange-act-assert)
    - [Type hints](#type-hints)
    - [Documentation](#documentation)
      - [Docstrings](#docstrings)
      - [Building the documentation](#building-the-documentation)
      - [Publishing the documentation](#publishing-the-documentation)
    - [Changelog](#changelog)
  - [Releasing](#releasing)

<!-- include_in_docs_from_here -->

## Introduction

Thank you for considering contributing to `clinlp`! This project is open source and community driven, so it is largely dependent on contributors like you. This document should help you get involved with the project, whether reporting a bug or issue, requesting a feature, making changes to the codebase, or contributing in any other way.

Please keep in mind that this page describes the ideal process and criteria for contributions. We understand that not all contributions will meet these criteria, and that is perfectly fine. We are happy to assist, so please don't hesitate to reach out to us.

## Contact

Our preferred way of communication is through [issues](https://github.com/umcu/clinlp/issues), GitHubs built-in issue tracker. We use it for most communication, including questions, bug reports, feature requests, help getting started, etc. This way, the entire community can benefit from the discussion. If this is not an option, you can also reach out to us by e-mail: [analytics@umcutrecht.nl](mailto:analytics@umcutrecht.nl).

To create an issue right now, you can use the following link: [Create an issue](https://github.com/umcu/clinlp/issues/new/choose).

We will try to respond to you as soon as possible. Please keep in mind that we are with a small group of maintainers, so we might not always be able to get back to you within a few days.

### Questions

If you have any questions about the project, or need help getting started, please include at least the following information in your issue:

- A clear and descriptive title
- A description of the question
- Any other relevant information

### Bug reports

If you encounter a bug, such as an error or unexpected behavior, please include at least the following information in your issue:

- A clear and descriptive title
- A detailed description of the bug
- Steps to reproduce the bug (if possible, include a minimal code snippet)
- Expected behavior
- Actual behavior
- The version of `clinlp` you are using
- The version of Python you are using
- Any other relevant information

### Feature requests and contributions

We will happily consider (ideas for) new additions to `clinlp`.

If you have a feature request that you would like someone to pick up, please include at least the following information in your issue:

- A clear and descriptive title
- A detailed description of the feature
- A use case for the feature
- Your contact information (if you would like to be involved in the development)
- Any other relevant information

Keep in mind that a feature request might not be picked up immediately, or at all. We will try to keep the roadmap up to date, so you can see what is being worked on, and what is planned for the future. Furthermore, remember that `clinlp` is a collection of generic components that process clinical text written in Dutch. If the proposed addition does not meet those criteria, a separate release might be a better option. We typically also don't include preprocessing components (e.g. fixing encodings, de-identification, etc.), as those should preferably be handled at the source.

If you would like to contribute to the project yourself directly, it's recommended to [create an issue](https://github.com/umcu/clinlp/issues/new/choose) to discuss your idea beforehand. This way, we can make sure that your contribution is in line with the project's goals and that it is not already being worked on by someone else. Of course, for small changes that only touch a couple of lines of code, you can also directly create a pull request. When you are ready to start working on your contribution, please follow the steps outlined in the [Pull requests](#pull-requests) section.

## Pull requests

If you would like to contribute to the project by making changes to the codebase, please follow these steps:

- Fork the repository
- Create a new branch for your changes
- Make your changes locally
  - Check your changes are in line with the project's [Coding Standards](#coding-standards)
  - Document your changes in `CHANGELOG.md`
- Push your changes to your fork
- Create a pull request
- Wait for feedback
- Address the feedback

Normally, the pull request will be merged by maintainers after all feedback has been addressed, after which it will be included in the next release.

## Roadmap

We like to keep the project roadmap open and transparent. You can find the roadmap in the [projects](https://github.com/orgs/umcu/projects/3). All issues are automatically added to the roadmap, so you can see what is being worked on, and what is planned for the future.

## Developer environment setup

You can setup a local development environment by cloning the repository:

```bash
git clone git@github.com:umcu/clinlp.git
cd clinlp
```

We use `uv` for managing dependencies, and building the package. If you do not have it yet, installation is covered in the [official `uv` guide](https://docs.astral.sh/uv/getting-started/installation/).

Then, you can install the project with dependencies using:

```bash
uv sync --all-extras
```

We use Poe the Poet for running common tasks, configured in `pyproject.toml`. It is automatically installed as a dev dependency. You can run Poe tasks using:

```bash
poe <task>
```

### Repository structure

The repository is structured as follows:

| Directory    | Description                                               |
| ------------ | --------------------------------------------------------- |
| `.github`    | GitHub actions and workflows                              |
| `docs`       | The documentation                                         |
| `media`      | Media files for the documentation or readme               |
| `scripts`    | Scripts for simple tasks that are not part of the package |
| `src/clinlp` | The source code for `clinlp`                              |
| `tests`      | The tests for `clinlp`                                    |

## Coding standards

With `clinlp` we aim for code that is production-ready. We have a few guidelines that we follow to ensure that the codebase maintains a high quality.

### General principles

Please keep the following principles in mind when writing code:

- Avoid repetitions, but refactor instead
- Keep it simple
- Only implement what is needed
- Keep functions small and focused
- Use descriptive names for variables, functions, classes, etc.
- Apply SOLID principles and design patterns where applicable
- Think about the user of your code, and make it easy to use
- Consider the maintainability and scalability of your code

We fully acknowledge that writing production ready code is a skill that takes time to develop. We are happy to work together, so please don't hesitate to reach out to us. This is especially true for scientific researchers who are working on something cool, but are new to software development.

### Creating a component

When creating a new component for `clinlp`, try to:

- Use a class to define the component, and use `__init__` to set the arguments.
- Inherit from `Pipe` to make it compatible with `spaCy`.
- Use the `clinlp_component` decorator, to automatically register it in the component library.
- Use a dictionary to define any defaults, and pass this to `default_config` of `clinlp_component`.
- Use type hints for all arguments and return values.
- Use the `requires` and `assigns` arguments to specify which fields the component needs, and which it sets.
- Implement the actual behavior of the component in the `__call__` method

The following code snippet shows an example of a new component:

```python
from clinlp.utils import clinlp_component
from spacy.language import Pipe
from spacy.tokens import Doc

_defaults = {
  "arg_1": 1,
  "arg_2": True
}

@clinlp_component(
  name="my_new_component",
  requires=["input_spacy_field"],
  assigns=["output_spacy_field"],
  default_config=_defaults
)

class MyNewComponent(Pipe):

  def __init__(self, arg_1: Type = _defaults['arg_1'], arg_2: Type = _defaults['arg_2']):
    ...

  def __call__(doc: Doc) -> Doc:
    ...
    return doc
```

### Formatting and linting

We use `ruff` for both formatting and linting. It is configured in `pyproject.toml`.

The `ruff` formatter is a drop-in replacement for `black`. You can run it using the following command, which will format the codebase:

```bash
poe format
```

The `ruff` linter checks for any common errors, that should be resolved before committing changes to the codebase. You can run it using the following command:

```bash
poe check
```

If any issues are found, some can be automatically fixed using the following command:

```bash
poe check --fix
```

### Dependencies

We use `uv` for managing dependencies. Adding a dependency is as straightforward as:

```bash
uv add <package>
```

To keep `clinlp` lightweight, we only include dependencies that are strictly necessary in the base package, and use optional dependencies for additional functionality (e.g. transformers, metrics). You can add an optional dependency using:

```bash
uv add --optional <optional_group> <package>
```

If you are adding specific development dependencies, please add them as such using:

```bash
uv add --dev <package>
```

### Tests

We use the `pytest` framework for testing. New code should preferably be accompanied by tests. We aim for a test coverage of at least 85%.

You can run the tests locally by running:

```bash
poe test
```

We preferably use the following `pytest` best practices:

- Use fixtures to share setup code
- Use parametrize to run the same test with different inputs
- Use marks to skip tests, or to run tests with specific marks

Additionally, we keep separation between unit, integration and regression tests:

- Unit tests should be fast and test a single unit of code. Each module in the codebase should at least have a corresponding module with unit tests.
- Integration tests should test the interaction between different components.
- Regression tests should test the performance of one or more components on real text examples. It's useful to also include some test cases that don't produce the correct result and mark them as known failure for future improvement.

If any test data is required for your tests, please add it to the `tests/test_data` directory. If possible, use a text-based format such as JSON or CSV, or if that's not possible at least an open format is preferred.

#### Arrange, act, assert

We use the arrange, act and assert pattern to structure our tests, which helps to make tests more readable and maintainable. In this pattern, each test is divided into three steps:

- The arrange step sets up the conditions for the test
- The act step performs the action that is being tested
- The assert step verifies that the action has the expected result

In the codebase, we make these steps explicit by using comments:

```python
def test_some_function():
    # Arrange
    ...

    # Act
    ...

    # Assert
    ...
```

Or, alternatively:

```python
import pytest

# Arrange
@pytest.fixture()
def my_fixture():
    ...

def test_some_function(my_fixture):
    # Act
    ...

    # Assert
    ...
```

If it's not possible to separate these steps, consider refactoring either the code or the test. In some cases, it might be necessary to combine steps, but this should be the exception rather than the rule.

### Type hints

We use type hints throughout the codebase, for both functions and classes. This helps with readability and maintainability. Usage of type hints is enforced by `ruff`.

### Documentation

We like our code to be well documented. The documentation can be found in the `docs` directory. If you are making changes to the codebase, please make sure to update the documentation accordingly. If you are adding new components, please add them to the [component library](https://clinlp.readthedocs.io/en/latest/components.html), and following the existing structure.

#### Docstrings

We include docstrings for all modules, classes and functions, specifically using the NumPy docstring format. You can find more information about this format [here](https://numpydoc.readthedocs.io/en/latest/format.html). The exact format should become clear from other docstrings in the codebase.

Each docstring should start with a summary line, optionally followed by more explanation. Please make sure that the summary line completes the following sentence:

| Type        | Format                                                 |
| ----------- | ------------------------------------------------------ |
| `Module`    | This module contains... `summary line`                 |
| `Class`     | This class represents (the/a)... `summary line`        |
| `Function`  | When you call this function, it will... `summary line` |

Note that docstrings are used to automatically generate the API, which is also publicly available. Only use inline comments if absolutely necessary to clear up unavoidably confusing code.

#### Building the documentation

We use `sphinx` for generating the documentation pages. If you want to build the documentation locally, you need to install clinlp:

```bash
uv sync --all-extras
```

Then, you can build the documentation by running:

```bash
poe build_docs
```

You should find the docs in `html/_build`.

#### Publishing the documentation

On changes to `main`, documentation is automatically built and published on [https://clinlp.readthedocs.io/](https://clinlp.readthedocs.io/). This setup is configured in `docs/.readthedocs.yml`.

### Changelog

Please make sure to update the `CHANGELOG.md` file with a description of your changes, at least every time a PR is created. This file should be updated with every change that is merged to `main`. Instructions on how to format the changelog can be found in the file itself.

## Releasing

> **Note:** Only maintainers can release new versions of `clinlp`.

To prepare a new release, you can increment the version number manually in `pyproject.toml`.

Please update `CHANGELOG.md` with the version number and date of this release. If everything went well, all changes that were merged to `main` should already be documented. But it's always good to double check.

Next, create a [new release](https://github.com/umcu/clinlp/releases) on GitHub. Please use the version number, in the exact format `v0.0.0` as the tag and the title. In the description box, copy the changes from `CHANGELOG.md`.

After creating the release, a [GitHub action](https://github.com/umcu/clinlp/actions) will automatically build and publish the package to PyPI. Please verify that it completed successfully, and the correct version is indeed visible on [PyPI](https://pypi.org/project/clinlp/).

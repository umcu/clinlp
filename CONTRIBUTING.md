# Contributing to `clinlp`

## Introduction

Thank you for considering contributing to `clinlp`! This project is intended to be open source and community driven, so it is largely dependent on contributors like you. This document is intended to help you get started with contributing to the project, whether reporting a bug or issue, requesting a feature, making changes to the codebase, or contributing in any other way.

Please keep in mind that this page describes the ideal process and criteria for contributions. We understand that not all contributions will meet these criteria, and that is perfectly fine. We are happy to assist, so please don't hesitate to reach out to us.

### Contact

Our preferred way of communication is through [issues](https://github.com/umcu/clinlp/issues), GitHubs built-in issue tracker. This way, the entire community can benefit from the discussion. If this is not an option, you can also reach out to us by e-mail: [analytics@umcutrecht.nl](mailto:analytics@umcutrecht.nl).

## Issues

### Bug reports

If you encounter a bug, please create an issue. Make sure to include as much information as possible, such as:

- A clear and descriptive title
- A detailed description of the bug
- Steps to reproduce the bug (if possible, include a minimal code snippet)
- Expected behavior
- Actual behavior
- Screenshots, if applicable
- Any other relevant information
- If possible, include the version of `clinlp` you are using
- If possible, include the version of Python you are using

### Feature requests

If you have a feature request, please create an issue. Make sure to include as much information as possible, such as:

- A clear and descriptive title
- A detailed description of the feature
- A use case for the feature
- Any other relevant information

### Contributions

If you would like to contribute to the project, please create an issue to discuss your idea. This way, we can make sure that your contribution is in line with the project's goals and that it is not already being worked on by someone else.

### Should it be in `clinlp`?

Please keep in mind that `clinlp` is intended to be a collection of generic components that process clinical text written in Dutch. If your contribution does not meet those criteria, a separate release might be a better option.

## Pull requests

If you would like to contribute to the project by making changes to the codebase, please follow these steps:

- Fork the repository
- Create a new branch for your changes
- Make your changes
- Check your changes are in line with the project's [coding standards](#coding-standards)
- Commit your changes
- Push your changes to your fork
- Create a pull request
- Wait for feedback
- Address the feedback

Normally, the pull request will be merged by maintainers after all feedback has been addressed, after which it will be included in the next release.

## Roadmap

We like to keep the project roadmap open and transparent. You can find the roadmap in the [projects](https://github.com/orgs/umcu/projects/3). All issues are automatically added to the roadmap, so you can see what is being worked on, and what is planned for the future.

## Setting up the developer environment

You can setup a local development environment by cloning the repository:

```bash
git clone git@github.com:umcu/clinlp.git
cd clinlp
```

We use poetry for managing dependencies, and building the package. If you do not have it yet, it's recommended to follow the [official installation guide](https://python-poetry.org/docs/#installation).

Then, you can install the dependencies by running:

```bash
poetry install . --group dev 
```

## Coding standards

With `clinlp` we aim for code that is production-ready, well tested, scalable, maintainable, etc. We have a few guidelines that we follow to ensure that the codebase maintains a high quality.

### General principles

Please keep the following principles in mind when writing code:

- Avoid repetitions, but refactor instead
- Keep it simple, but not simpler
- Only implement what is needed
- Keep functions small and focused
- Use descriptive names for variables, functions, classes, etc.
- Apply SOLID principles and design patterns where applicable

We fully acknowledge that writing production ready code is a skill that takes time to develop. We are happy to help work together, so please don't hesitate to reach out to us. This is especially true for scientific researchers who are new to software development.

### Formatting and linting

We use `ruff` for both formatting and linting. It is configured in `pyproject.toml`.

The `ruff` formatter is a drop-in replacement for `black`. You can run it using the following command, which will format the codebase:

```bash
ruff format
```

The `ruff` linter checks for any common errors, that should be resolved before committing changes to the codebase. You can run it using the following command:

```bash
ruff lint
```

If any issues are found, some can be automatically fixed using the following command:

```bash
ruff lint --fix
```

### Tests

We use the `pytest` framework for testing. New code should preferably be accompanied by tests. We aim for a test coverage of at least 80%.

You can run the tests locally by running:

```bash
pytest .
```

We preferably use the following `pytest` best practices:

- Use fixtures to share setup code
- Use parametrize to run the same test with different inputs
- Use marks to skip tests, or to run tests with specific marks

#### Arrange, act, assert

Try to write your tests according to the arrange, act and assert pattern:

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

### Type hints

We use type hints throughout the codebase, for both functions and classes. This helps with readability and maintainability. We aim to use type hints for all functions and classes. This is also enforced by `ruff`.

### Documentation

Please ensure that your code is well documented. You can find the documentation in the `docs` directory. If you are making changes to the codebase, please make sure to update the documentation accordingly.

#### Docstrings

Please make sure to include docstrings for all modules, classes and functions. We use the NumPy docstring format. You can find more information about this format [here](https://numpydoc.readthedocs.io/en/latest/format.html).

Each docstring should start with a summary line, optionally followed by more explanation. Please make sure that the summary line completes the following sentence:

| Type        | Summary line                            |
| ----------- | --------------------------------------- |
| `Modules`   | This module contains...                 |
| `Classes`   | This class represents (the/a)...        |
| `Functions` | When you call this function, it will... |

#### Building the documentation

We use `sphinx` for generating the documentation pages. If you want to build the documentation locally, you need to install the documentation dependencies:

```bash
poetry install . --group docs
```

Then, you can build the documentation by running:

```bash
make build-docs
```

You should find the docs in `html/_build`.

#### Readthedocs.io

On changes to `main`, documentation is automatically built and published on [clinlp.readthedocs.io](https://clinlp.readthedocs.io/). This setup is configured in `docs/.readthedocs.yml`.

### Changelog

Please make sure to update the `CHANGELOG.md` file with a description of your changes. This file should be updated with every change that is merged to `main`.

## Releasing

To prepare a new release, you can increment the version number using the following command.

```bash
poetry version <major|minor|patch>
```

Please update `CHANGELOG.md` with the version number and date. If everything went well, all changes that were merged to `main` should already be documented. But it's always good to double check.

Next, create a [new release](https://github.com/umcu/clinlp/releases) on GitHub. Please use the version number as the tag, and the version number as the title. You can copy the changelog entry as the description.

After creating the release, a [GitHub action](https://github.com/umcu/clinlp/actions) will automatically build and publish the package to PyPI. Please verify that it completed successfully, and the correct version is indeed visible on [PyPI](https://pypi.org/project/clinlp/).

# Introduction

```{include} ../../README.md
:start-after: <!-- start_intro_line_1 -->
:end-before: <!-- start_intro_line_2 -->
```

`clinlp` is a Python library for performing NLP on clinical text written in Dutch. It is designed to be a standardized framework for building, maintaining and sharing solutions for NLP tasks in the clinical domain. The library is built on top of the [`spaCy`](https://spacy.io/) library, and extends it with components that are specifically tailored to Dutch clinical text. The library is open source and welcomes contributions from the community.

## Motivation

`clinlp` was motivated by the lack of standardized tools for processing clinical text in Dutch. This makes it difficult for researchers, data scientists and developers working with Dutch clinical text to build, validate and maintain NLP solutions. With `clinlp`, we aim to fill this gap.

## Principles

We organized `clinlp` around four basic principles: useful NLP tools, a standardized framework, production-ready quality, and open source collaboration.

### 1. Useful NLP tools

```{include} ../../README.md
:start-after: <!-- start_intro_line_2 -->
:end-before: <!-- start_intro_line_3 -->
```

There are many interesting NLP tasks in the clinical domain, like normalization, entity recognition, qualifier detection, entity linking, summarization, reasoning, and many more. In addition to that, each task can often be solved using rule-based methods, classical machine learning, deep learning, transformers, or a combination of these, with trade-offs between them.

The main idea behind `clinlp` is to build, maintain and share solutions for these NLP tasks, specifically for clinical text written in Dutch. In `clinlp`, we typically call a specific implementation for a task a "component". For instance: a rule-based sentence boundary detector, or a transformer-based negation detector.

Currently, `clinlp` mainly includes components used for information extraction, such as tokenizing, detecting sentence boundaries, normalizing text, detecting entities, and detecting qualifiers (e.g. negation, uncertainty). The library is regularly being updated with new or improved components, both components for different tasks (e.g. entity linking, summarization) and components that use a different method for solving a task (e.g. a transformer-based entity recognizer).

```{admonition} Contributing
:class: note

Components can be built by anyone from the Dutch clinical NLP field, typically a researcher, data scientist, engineer or clinician who works with Dutch clinical text in daily practice. If you have a contribution in mind, please check out the [Contributing](contributing) page.
```

When building new solutions, we preferably start with a component that implements a simple, rule-based solution, which can function as a baseline. Then subsequently, more sophisticated components can be built. If possible, we try to (re)use existing implementations, but if needed, building from scratch is also an option.

We prefer components to work out of the box, but to be highly customizable. For instance, our implementation of the [Context Algorithm](components.md#clinlp_context_algorithm) has a set of built in rules for for qualifying entities with Presence, Temporality and Experiencer properties. However, both the types of qualifiers and the rules can easily be modified or replaced by the user. This way, the components can be used in a wide variety of use cases, and no user is forced to use a one-size-fits-all solution.

```{admonition} Validating components
:class: important

Remember, there is no guarantee that components based on existing rules or pre-trained models also extend to your particular dataset and use case. It is always recommended to evaluate the performance of the components on your own data.
```

In addition to functional components, `clinlp` also implements some functionality for computing metrics. This is useful for evaluating the performance of the components, and for comparing different methods for solving the same task.

An overview of all components included in `clinlp` can be found on the [Components](components) page.

### 2. Standardized framework

```{include} ../../README.md
:start-after: <!-- start_intro_line_3 -->
:end-before: <!-- start_intro_line_4 -->
```

Some of the real power from `clinlp` comes from the fact that the different components it implements are organized in a standardized framework. This framework ensures that the components can be easily combined and that they can be used in a consistent way. This makes it easy to build complex pipelines that can effectively process clinical text.

We use the [`spaCy`](https://spacy.io/) library as the backbone of our framework. This allows us to leverage the power of `spaCy`'s NLP capabilities and to build on top of it. We have extended `spaCy` with our own domain-specific language defaults to make it easier to work with clinical text. In a pipeline, you can mix and match different `clinlp` components with existing `spaCy` components, and add your own custom components to that mix as well. For example, you could use the `clinlp` normalizer, the `spaCy` entity recognizer, and a custom built entity linker in the same pipeline without any issues.

```{admonition} Getting familiar with spaCy
:class: note

It's highly recommended to read [`spaCy` 101: Everything you need to know (~15 minutes)](https://spacy.io/usage/spacy-101) before getting started with `clinlp`. Understanding the basic `spaCy` framework will make working with `clinlp` much easier.
```

In addition to the `spaCy` framework, we have added some additional abstractions and interfaces that make building components easier. For instance, if you want to add a new component that detects qualifiers, it can make use of the `QualifierDetector` abstraction, and the `Qualifier` and `QualifierClass` classes. This way, the new component can easily be integrated in the framework, while the developer can focus on building a new solution.

Finally, by adopting a framework, we can easily build components that wrap a specific pre-trained model. The transformer-based qualifier detectors included in `clinlp` are good examples of this. These components wrap around pre-trained transformer models, but fit seamlessly into the `clinlp` framework. This way, we can easily add new components that use the latest and greatest in NLP research.

### 3. Production-ready quality

```{include} ../../README.md
:start-after: <!-- start_intro_line_4 -->
:end-before: <!-- start_intro_line_5 -->
```

`clinlp` can potentially serve many types of users, including researchers, data scientists, engineers and clinicians. One thing they all have in common, is that they would like to rely on the library to work as expected. Our goal is to build a library with the robustness and reliability required in production environments, i.e. real world environments. To ensure this, we employ various software development best practices, including:

* Proper system design by using abstractions, interfaces and design patterns (where appropriate)
* Formatting, linting and type hints for a clean, consistent and readable codebase
* Versioning and a changelog to track changes over time
* Optimizations for speed and scalability
* Structural management of dependencies and packaging
* Extensive testing to ensure that the library works (and keeps working) as expected
* Documentation to explain the library's principles, functionality and how to use it
* Continuous deployment and frequent new releases

We actively maintain the library, and are always looking for ways to improve it. If you have suggestions how to further increase the quality of the library, please let us know.

More detail on the `clinlp` development practices can be found in the [Coding Standards](contributing.md#coding-standards) section of the contributing page.

### 4. Open source collaboration

```{include} ../../README.md
:start-after: <!-- start_intro_line_5 -->
:end-before: <!-- end_intro_lines -->
```

`clinlp` is being built as a free and open source library, but we cannot do it alone. As an open source project, we highly welcome contributions from the community. We believe that open source collaboration is the best way to build high quality software that can be used by everyone. We encourage you to contribute to the project by reporting issues, suggesting improvements, or even submitting your own code.

In order to be transparent, we prefer to communicate through means that are open to everyone. This includes using GitHub for issue tracking, pull requests and discussions, and using the `clinlp` documentation for explaining the library's principles and functionality. We keep our [Roadmap](roadmap) and [Changelog](changelog) up to date, so you can see what we are working on and what has changed in the library.

Finally, by working together in `clinlp`, we hope to strengthen the connections in our specific field of Dutch clinical NLP across organizations and institutions. By committing to making algorithms and implementations available in this package, and to collaboratively further standardize algorithms and protocols, we can ensure that the research is reproducible and that the algorithms can be used by others. This way, we can build on each other's work, and make the field of Dutch clinical NLP stronger.

## About

`clinlp` was initiated by a group of data scientists and engineers from the UMCU, who ran into practical issues working with clinical text and decided to build a library to solve them.

The library is currently actively maintained by:

* [Vincent Menger, ML engineer, UMCU](https://github.com/vmenger)
* [Bram van Es, Assistant Professor, UMCU](https://github.com/bramiozo)

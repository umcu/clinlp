lint:
	python -m ruff check .

clean:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf dist

build-docs:
	sphinx-apidoc --module-first --force --templatedir=docs/templates -o docs/source/api src
	sphinx-build docs/source docs/_build/html -c docs/

.PHONY: lint clean build-docs

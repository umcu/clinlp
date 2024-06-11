build-docs:
	sphinx-apidoc --module-first --force --templatedir=docs/_templates -o docs/source/api/clinlp src
	sphinx-build docs/source docs/_build/html -c docs/
	python docs/emojize.py docs/_build/html	

.PHONY: build-docs

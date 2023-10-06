all:
	python -m pytest
	python -m black .
	python -m isort .

.PHONY: all

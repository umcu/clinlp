lint:
	python -m black .
	python -m isort .
	python -m flake8 .

.PHONY: lint

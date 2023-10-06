lint:
	python -m black .
	python -m isort .
	python -m flake8 .

clean:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf dist

.PHONY: lint

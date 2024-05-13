lint:
	python -m ruff check .

clean:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf dist

.PHONY: lint clean

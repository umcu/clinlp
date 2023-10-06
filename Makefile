MAX_LINE_LENGTH := 88
CHECK ?= 0

format_dirs := clinlp/ tests/
lint_dirs := clinlp/

black_args := --line-length $(MAX_LINE_LENGTH)
isort_args := --profile black
docformatter_args := --recursive --wrap-summaries $(MAX_LINE_LENGTH) --wrap-descriptions $(MAX_LINE_LENGTH) --pre-summary-newline

typehints_args := --select ANN001,ANN2,ANN3 --max-line-length $(MAX_LINE_LENGTH)
doclint_args := --disable=all --enable C0112,C0115,C0116
pylint_args := --disable=C0112,C0114,C0115,C0116 --max-line-length=$(MAX_LINE_LENGTH)

ifeq ($(CHECK), 1)
	black_args += --check
	isort_args += -c
	docformatter_args := --check $(docformatter_args)
	doclint_args += --fail-under 10.0
	pylint_args += --fail-under 9.0

else
	docformatter_args := --in-place $(docformatter_args)
	typehints_args += --exit-zero
	doclint_args += --exit-zero
	pylint_args += --exit-zero

endif

format: black isort

lint: typehints pylint

black:
	python -m black $(black_args) $(format_dirs)

isort:
	python -m isort $(isort_args)  $(format_dirs)

typehints:
	python -m flake8 $(typehints_args) $(lint_dirs)

pylint:
	python -m pylint $(pylint_args) $(lint_dirs)

test:
	python -m pytest --cov-report html --cov-report lcov --cov=clinlp --cov-fail-under=80 tests/

clean:
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf coverage.lcov
	rm -rf .pytest_cache
	rm -rf dist

.PHONY: format lint black isort docformat typehints doclint pylint test clean

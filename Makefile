SHELL=/bin/bash
LINT_PATHS=sbx/ tests/ setup.py

pytest:
	./scripts/run_tests.sh

type:
	pytype -j auto

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

ruff:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source --line-length 127
	# exit-zero treats all errors as warnings.
	ruff ${LINT_PATHS} --exit-zero --line-length 127


format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format type lint

# doc:
# 	cd docs && make html

# spelling:
# 	cd docs && make spelling

# clean:
# 	cd docs && make clean

# PyPi package release
release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: clean spelling doc lint format check-codestyle commit-checks

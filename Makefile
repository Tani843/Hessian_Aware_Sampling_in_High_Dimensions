.PHONY: install test benchmark docs clean lint format

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest --cov=src tests/ --cov-report=html

test-integration:
	pytest tests/test_integration.py -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ examples/

# Benchmarking and experiments
benchmark:
	python3 examples/comprehensive_benchmark.py

experiment:
	python3 scripts/run_complete_experiment.py

quick-test:
	python3 scripts/run_complete_experiment.py --quick-test

# Documentation
docs:
	cd docs/jekyll_site && bundle exec jekyll build

docs-serve:
	cd docs/jekyll_site && bundle exec jekyll serve

# Cleanup
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage
	rm -rf results/temp/ test_outputs/

# Release
release: clean test lint
	python setup.py sdist bdist_wheel
	# twine upload dist/*

# Complete pipeline
all: install-dev format lint test benchmark docs
	@echo "✅ Complete pipeline finished successfully!"
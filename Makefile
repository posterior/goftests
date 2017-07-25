.phony: test

test:
	find . -type f -name \*.py -exec flake8 {} \;
	python -m unittest discover
	@echo '------------'
	@echo 'PASSED TESTS'

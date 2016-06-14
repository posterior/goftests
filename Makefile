.phony: test

test:
	find . -type f -name \*.py -exec flake8 {} \;
	nosetests -v goftests
	@echo '------------'
	@echo 'PASSED TESTS'

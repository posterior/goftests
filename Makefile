.phony: test

test:
	find . | grep '\.py$$' | xargs pep8
	find . | grep '\.py$$' | xargs pyflakes
	nosetests -v goftests
	@echo '------------'
	@echo 'PASSED TESTS'

# :copyright: 2025 Jakub Res
# :license: MIT
# :author: Matej Olexa <olexa.matej@gmail.com>
# :author: Jakub Res <iresj@fit.vut.cz>


.PHONY: lint test
lint:
	python3 -m ruff check src tests
test:
	python3 -m pytest -q

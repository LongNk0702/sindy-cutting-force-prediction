.PHONY: setup lint test run-demo

setup:
\tpython -m venv .venv
\t. .venv/bin/activate && pip install -r requirements.txt
\tpre-commit install

lint:
\tblack src tests
\tisort src tests
\tflake8 src tests

test:
\tpytest -q

run-demo:
\tpython examples/cli_demo.py

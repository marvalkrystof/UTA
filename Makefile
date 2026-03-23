.PHONY: install install-dev test test-backend test-frontend frontend frontend-deps examples

# Install uta_solver package
install:
	pip install .

# Install uta_solver in development (editable) mode
install-dev:
	pip install -e .

# Run all tests
test: install-dev test-backend test-frontend

# Run uta_solver tests
test-backend:
	python -m pytest uta_solver/tests/ -v

# Run frontend tests
test-frontend:
	python -m pytest frontend/tests/ -v

# Install frontend dependencies
frontend-deps:
	pip install -e ".[frontend]"

# Run Streamlit frontend
frontend: frontend-deps
	cd frontend && streamlit run app.py \
		--server.runOnSave true \
		--server.fileWatcherType poll

# Run all examples
examples: install
	python -m examples.apartments
	python -m examples.cars

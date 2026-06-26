# :copyright: 2025 Jakub Res
# :license: MIT
# :author: Matej Olexa <olexa.matej@gmail.com>
# :author: Jakub Res <iresj@fit.vut.cz>

CONDA_ENV="llms"
PYTHON_FILES=src tests scripts huggingface-scraper rome_benchmark.py

.PHONY: install setup mkdir dirs check-rome lint format

check-rome:
	git diff --exit-code main -- src/rome src/handlers/rome.py src/config/config.yaml src/config/model

lint:
	ruff check $(PYTHON_FILES)

format:
	ruff format $(PYTHON_FILES)

# Run setup scripts to install the toolset
install:
	@if ! conda info --envs | grep -q "^$(CONDA_ENV)\s"; then \
		echo "Creating conda environment $(CONDA_ENV)"; \
		conda create -n $(CONDA_ENV) -y; \
		conda run -n $(CONDA_ENV) bash conda_install.sh \
	else \
		echo "Conda environment $(CONDA_ENV) already exists"; \
	fi

# Make directories
dirs: mkdir

mkdir:
	mkdir -p models
	mkdir -p datasets
	mkdir -p notebooks
	mkdir -p analysis_out
	mkdir -p prefix_cache
	mkdir -p data
	mkdir -p data/evals
	mkdir -p data/second_moment_stats
	mkdir -p data/causal_trace_stats
	mkdir -p data/new_weights
	mkdir -p data/figs

# Setup the environment for the project
setup: install mkdir

CONDA_ENV="llms"

.PHONY: install activate causal_trace

# Run setup scripts to install the toolset
install:
	@if ! conda info --envs | grep -q "^$(CONDA_ENV)\s"; then \
		echo "Creating conda environment $(CONDA_ENV)"; \
		conda create -n $(CONDA_ENV) -y; \
		conda run -n $(CONDA_ENV) bash conda_install.sh \
	else \
		echo "Conda environment $(CONDA_ENV) already exists"; \
	fi

# Make dirrectories
mkdir:
	mkdir -p models
	mkdir -p datasets
	mkdir -p notebooks
	mkdir -p data
	mkdir -p data/evals
	mkdir -p data/second_moment_stats
	mkdir -p data/causal_trace_stats
	mkdir -p data/new_weights
	mkdir -p data/figs

# Setup the environment for the project
setup: install mkdir
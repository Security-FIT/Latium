CONDA_ENV="llms"
.PHONY: install activate causal_trace

# Run setup scripts to install the toolset
install:
	@if ! conda info --envs | grep -q "^$(CONDA_ENV)\s"; then \
		conda create -n $(CONDA_ENV) -y; \
	else
		@echo "Conda environment $(CONDA_ENV) already exists" \
		@exit 2
	fi
	conda run -n $(CONDA_ENV) bash conda_install.sh

# Activate the conda env
activate:
	conda activate $(CONDA_ENV)

# Run causal_trace from rome with passed parameters from the make run command
causal_trace: 
	python -m reimagined.rome.causal_trace.causal_trace $(ARGS)
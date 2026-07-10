# ROME internals

ROME code is split by responsibility:

- `prefixes.py`: prefix modes, generation, and cache files.
- `subjects.py`: subject-token lookup.
- `optimization.py`: key gathering, value optimization, and weight insertion.
- `activations.py`: activation-shape normalization and covariance contributions.
- `covariance.py`: second-moment batching, caching, and loading.

`common.py` is a compatibility facade for historical imports. New code should
import the focused module that owns the operation.

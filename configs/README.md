# Configuration policy

Configurations are versioned research contracts, not ad-hoc command flags.

- `benchmark/`: splits, episode construction, horizons, metrics, storage limits
- `models/`: authorized model definitions only
- `experiments/`: run-level composition of benchmark/model settings

Every formal run records the exact config path and SHA-256 in its immutable run
manifest. Do not mutate a config after using it for a completed run; create a
new versioned file instead.



## Quick Start

### Initialize the workspace:
   ```bash
   workspace-init
   ```
   This creates a `~/.blancops_profile` pointer and sets up the default configuration and data directories.


### Training a Model

```bash
# Generate training lookup tables
construct-train-lookups --fits_path <path/to/train/data/in/fits/format> --outdir <path/to/lookup/outdir>

# Run training
run-train --config <path/to/config>
```
There is a template train config file in `configs/template_train_config.json`

### Running Validation/Evaluation

```bash
# Validate a trained model
run-validate --model-dir deployable_models/bc_v0

# Run prediction/simulation
run-simulate --model-dir deployable_models/bc_v0
```


<!-- ### CLI Entry Points

All CLI commands are defined in `pyproject.toml` and map to functions in `blancops/scripts/`:

| Command | Purpose |
|---------|---------|
| `workspace-init` | Initialize workspace configuration and data |
| `construct-train-lookups` | Pre-compute training lookup tables |
| `run-train` | Train an RL policy |
| `run-validate` | Validate/evaluate a trained model |
| `run-simulate` | Run prediction/simulation |
| `run-live-scheduler` | Start the Live Scheduler |

Alternatively, invoke directly:
```bash
python -m blancops.scripts.train --help
``` -->

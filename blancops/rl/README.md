# Training and Validation Models

## Initialize the workspace

```bash
workspace-init
```

This creates a `~/.blancops_profile` pointer and sets up the default configuration and data directories.

## Training a Model

```bash
# Generate training lookup tables
build-train-lookups --fits_path <path/to/train/data/in/fits/format> --outdir <path/to/lookup/outdir>

# Run training
run-train -c <path/to/config>
# or, alternatively
run-train --config <path/to/config>
```

The training routine saves training results and best model in the following structure: 
<!-- in the directory, `<config/specified/parent/dir>/run_<YYMMdd>_<HHmmss>` with the following structure: -->

```
├─ experiment_dir/
│  ├─ run_<YYMMdd_HHMMSS>
│     ├─ configs/
│        ├─ checkpoint_epoch_<epoch_num>_metric_<metric_val>.pt
│        ├─ checkpoint_history.json
│        ├─ latest_checkpoint.pt
│        ├─ model.pt
│        ├─ normalization_stats.json
│     ├─ metrics/
│     ├─ checkpoints/
│     ├─ logs/
│     ├─ figures/
checkpoint_epoch_030_metric_8.3777.pt  checkpoint_history.json  latest_checkpoint.pt  model.pt  normalization_stats.json
```

- checkpoints
   - 
- configs
- figures
- logs
- metrics

<!-- There is a template train config file in `configs/template_train_config.json` -->
<!-- 
## Running Validation/Evaluation

```bash
# Validate a trained model
run-validate -t <path/to/trained/model/dir> 
# Run prediction/simulation
run-simulate --model-dir deployable_models/bc_v0
``` -->


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

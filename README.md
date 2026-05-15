
# BlancOps
This repo con is a reinforcement-learning-based autonomous scheduling agent for the BLANCO telescope. It operates in two primary modes: **offline training/evaluation** for developing scheduling policies, and the **Live Scheduler** for real-time, human-in-the-loop autonomous observation scheduling with live telescope telemetry.

For instructions on deploying the real-time observation scheduling agent, see the [live scheduler documentation](./blancops/live_scheduler/README.md).

## Quick Start

### Setup

TODO

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

<!-- ### Running Validation/Evaluation

```bash
# Validate a trained model
run-validate --model-dir deployable_models/bc_v0

# Run prediction/simulation
run-simulate --model-dir deployable_models/bc_v0
``` -->

### Real-Time Scheduling

```bash
run-live-scheduler
```

Or directly:
```bash
python scripts/run_live_scheduler.py
```

For detailed Live Scheduler documentation, see [live scheduler README](./blancops/live_scheduler/README.md).


<!-- ## Project Structure

```
blancops/
├── __init__.py
├── blanco/                 # BLANCO-specific implementations
├── configs/                # Configuration templates (packaged)
├── data/                   # Data files and lookups
│   ├── train/             # Training data
│   ├── fits/              # FITS data files
│   └── lookups/           # Pre-computed lookup tables
├── data_quality/          # Data validation and filtering
├── environment/           # RL environment definitions
├── ephemerides/           # Ephemeris calculations
├── evaluation/            # Offline evaluation tools
├── io/                    # I/O utilities
├── live_scheduler/        # Real-time scheduling system
│   ├── README.md          # Live Scheduler documentation
│   ├── orchestrator.py    # Central control loop
│   ├── client.py          # Telescope hardware interface
│   ├── model_runner.py    # ML inference engine
│   ├── interface.py       # User interface
│   └── progress_manager.py # Session tracking
├── math/                  # Mathematical utilities
├── plotting/              # Visualization tools
├── rl/                    # Reinforcement learning framework
├── scripts/               # CLI entry points
│   ├── init.py           # Workspace initialization
│   ├── generate_train_lookups.py
│   ├── train.py          # Model training
│   ├── validate.py       # Model validation
│   ├── predict.py        # Prediction/simulation
│   └── run_live_scheduler.py
├── telescope/             # Telescope control and telemetry
└── utils/                 # General utilities

tests/                      # Test suite
├── unit/                  # Unit tests
└── integration/           # Integration tests

deployable_models/         # Trained model weights
├── bc_v0/                # Simple production model
└── bc_v0_simple/         # Test model

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

<!-- ## References

- [Live Scheduler Documentation](./blancops/live_scheduler/README.md) - Detailed guide for real-time observation scheduling
- [BlancOps Configuration Guide](./blancops/configs/) - Configuration templates and defaults -->

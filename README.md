Code for a reinforcement learning based agent capable of optimizing telescope scheduling at BLANCO.

Before running a model, the following command must be run:

```model-init```

This (1) initializes the workspace (2) writes the `global_config.json` and a `template_train_config.json` to `blancops/configs` and (3) constructs and saves the training data lookup tables in `blancops/data/train`. By default, it assumes the workspace is in `blancops` and saves a pointer in ~/.blancops_profile. The train fits file is assumed to be at `blancops/data/train/decam-exposures-20251211.fits`

A simple behavior cloning agent can be trained by running

```model-train -c <path/to/config/file>```

The trained model can be evaluated for a validation night (e.g., for date 2017/01/05):

```model-eval -t <path/to/train/dir/containing/best/weights> -y 2017 -m 1 -d 5```

and results will be saved in a directory in the trained model directory


For instructions on deploying the real-time observation scheduling agent, see the [live scheduler documentation](./blancops/live_scheduler/README.md).

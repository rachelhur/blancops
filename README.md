
This repo contains a reinforcement-learning-based autonomous scheduling agent for the BLANCO telescope. It operates in two primary modes: **offline training/evaluation** for developing scheduling policies, and the **Live Scheduler** for real-time, human-in-the-loop autonomous observation scheduling with live telescope telemetry.


## Setup

<!-- conda environment setup -->
<!-- pip install blancops -->

For instructions on training and evaluating policies, or rolling out policies for a future night, see the [training and evaluating documentation](./blancops/rl/README.md).

For instructions on deploying the real-time observation scheduling agent, see the [live scheduler documentation](./blancops/live_scheduler/README.md).

<!-- ## References

- [Live Scheduler Documentation](./blancops/live_scheduler/README.md) - Detailed guide for real-time observation scheduling
- [BlancOps Configuration Guide](./blancops/configs/) - Configuration templates and defaults -->

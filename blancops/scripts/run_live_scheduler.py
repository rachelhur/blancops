"""Command-line entrypoint for the live scheduler."""

import argparse
import sys
import os
from pathlib import Path
import yaml

# Ensure blancops is in the path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blancops.live_scheduler.api_client import BlancoTelescopeAPI, MockTelescopeAPI
from blancops.live_scheduler.model_runner import MockModelRunner, AIModelRunner
from blancops.live_scheduler.interface import CLIInterface
from blancops.live_scheduler.state_manager import StateManager
from blancops.live_scheduler.orchestrator import SchedulerOrchestrator


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "live_scheduler_default.yaml"
)


def load_yaml_defaults(config_path):
    """Load scheduler defaults from a YAML config file."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file at {config_path} does not exist.")

    with config_path.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}

    if not isinstance(defaults, dict):
        raise ValueError(
            f"Config file at {config_path} must contain a mapping of defaults."
        )

    return defaults


def parse_args():
    """Parse CLI arguments, using YAML defaults unless overridden on the command line."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file containing scheduler defaults.",
    )

    pre_args, _ = pre_parser.parse_known_args()
    defaults = load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the blancops live scheduler.",
    )
    parser.add_argument(
        "--mode",
        choices=("mock", "blanco"),
        default=defaults.get("mode", "mock"),
        help="Scheduler mode to run.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.get("chunk_size", 3),
        help="Number of observations to generate per proposal chunk.",
    )
    parser.add_argument(
        "--mock-exposure-duration",
        type=float,
        default=defaults.get("mock_exposure_duration", 90),
        help="Simulated exposure duration in seconds when running in mock mode.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=defaults.get("model_path", "default_model.pt"),
        help="Path to the trained model used in blanco mode.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=defaults.get("output_directory", "./observing_logs"),
        help="Directory where observing logs are written.",
    )
    parser.add_argument(
        "--observing-poll-rate-sec",
        type=float,
        default=defaults.get("observing_poll_rate_sec", 1),
        help="How often to poll the telescope while waiting for an exposure.",
    )
    parser.add_argument(
        "--telemetry-poll-rate-sec",
        type=float,
        default=defaults.get("telemetry_poll_rate_sec", 20),
        help="How often to re-check telemetry before deciding whether to replan.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=defaults.get("session_id", None),
        help="Optional observing-session identifier.",
    )

    return parser.parse_args()


def main():
    """Run the live scheduler with YAML defaults and optional CLI overrides."""

    args = parse_args()
    print("Initializing blancops Live Scheduler...")

    # initialize components for running the scheduler
    if args.mode == "mock":
        api = MockTelescopeAPI(exposure_duration=args.mock_exposure_duration)
        model = MockModelRunner()
    else:
        api = BlancoTelescopeAPI()
        model = AIModelRunner(model_path_or_alias=args.model_path)
    ui = CLIInterface()
    state = StateManager(output_dir=args.output_directory, session_id=args.session_id)
    orchestrator = SchedulerOrchestrator(
        api,
        model,
        ui,
        state,
        chunk_size=args.chunk_size,
        observing_poll_rate_sec=args.observing_poll_rate_sec,
        telemetry_poll_rate_sec=args.telemetry_poll_rate_sec,
    )

    # run the scheduler loop
    try:
        orchestrator.run()  # XXX make sure there are stops built in there

    # Big Red Stop Button: handle graceful shutdown on Ctrl+C keyboard interrupt
    except KeyboardInterrupt:
        print("\n\n" + "!" * 88)
        print("EMERGENCY STOP TRIGGERED (Ctrl+C)")
        print("Halting all scheduler loops.")
        print("Ensuring telescope queue is cleared (XXX Placeholder).")
        print("!" * 88)
        sys.exit(0)


if __name__ == "__main__":
    main()

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
from blancops.math import units


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

    # ==================================================================================
    # API Settings
    # ==================================================================================
    api_group = parser.add_argument_group("API Settings")
    api_group.add_argument(
        "--api-mode",
        choices=("mock", "blanco"),
        default=defaults.get("api_mode", "mock"),
        help='Scheduler mode to run. Choose "mock" for offline testing.',
    )
    api_group.add_argument(
        "--mock-exposure-duration",
        type=float,
        default=defaults.get("mock_exposure_duration", 90),
        help="Simulated exposure duration in seconds when running in mock mode.",
    )

    # ==================================================================================
    # UI Settings
    # ==================================================================================
    ui_group = parser.add_argument_group("UI Settings")
    ui_group.add_argument(
        "--show-plots",
        action="store_true",
        default=defaults.get("show_plots", True),
        help="Whether to display proposed chunks interactively or just save to disk.",
    )
    ui_group.add_argument(
        "--ui-mode",
        choices=("cli", "web"),
        default=defaults.get("ui_mode", "cli"),
        help=(
            'User interface to use. Choose "cli" for command-line or "web" for local '
            "web interface."
        ),
    )

    # ==================================================================================
    # Model Settings
    # ==================================================================================
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model-path-or-alias",
        type=str,
        default=defaults.get("model_path_or_alias", "mock"),
        help='Alias name or path to the trained model. Choose "mock" for debugging.',
    )
    model_group.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "cpu"),
        help="Torch device for model inference, e.g. cpu, cuda, or cuda:0.",
    )
    model_group.add_argument(
        "--field-choice-method",
        choices=("interp",),
        default=defaults.get("field_choice_method", "interp"),
        help="Method used to map model outputs to physical fields.",
    )

    # ==================================================================================
    # Pathing
    # ==================================================================================
    path_group = parser.add_argument_group("Pathing")
    path_group.add_argument(
        "--field-lookup-dir",
        type=str,
        default=defaults.get("field_lookup_dir", "./field_lookups"),
        help="Directory containing field lookup tables for AI inference.",
    )
    path_group.add_argument(
        "--fields-path",
        type=str,
        default=defaults.get("fields_path", None),
        help=(
            "Optional path to a fields file used to construct lookups. If omitted, "
            "lookups are loaded from --field-lookup-dir."
        ),
    )
    path_group.add_argument(
        "--output-directory",
        type=str,
        default=defaults.get("output_directory", "./observing_logs"),
        help="Directory where observing logs are written.",
    )
    path_group.add_argument(
        "--session-id",
        type=str,
        default=defaults.get("session_id", None),
        help="Optional observing-session identifier.",
    )

    # ==================================================================================
    # Operational Settings
    # ==================================================================================
    operational_group = parser.add_argument_group("Operational Settings")
    operational_group.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.get("chunk_size", 3),
        help="Number of observations to generate per proposal chunk.",
    )
    operational_group.add_argument(
        "--min-chunk-size",
        type=int,
        default=defaults.get("min_chunk_size", None),
        help=(
            "Minimum number of observations allowed in a chunk before the scheduler "
            "automatically generates a new proposed chunk. Default None generates a "
            "new chunk immediately after every observation submission."
        ),
    )
    operational_group.add_argument(
        "--observing-poll-rate-sec",
        type=float,
        default=defaults.get("observing_poll_rate_sec", 1),
        help="How often to poll the telescope while waiting for an exposure.",
    )
    operational_group.add_argument(
        "--telemetry-poll-rate-sec",
        type=float,
        default=defaults.get("telemetry_poll_rate_sec", 20),
        help="How often to re-check telemetry before deciding whether to replan.",
    )

    # ==================================================================================
    # Observing Limits
    # ==================================================================================
    limits_group = parser.add_argument_group("Observing Limits")
    limits_group.add_argument(
        "--start-time",
        type=str,
        default=defaults.get("start_time", None),
        help=(
            "Optional start time. Both this and sun elevation conditions must be met "
            "to start the observing session."
        ),
    )
    limits_group.add_argument(
        "--start-sun-elevation-deg",
        type=float,
        default=defaults.get("start_sun_elevation_deg", -14),
        help=(
            "Optional sun elevation threshold. Both this and start time must be met "
            "to start the observing session."
        ),
    )
    limits_group.add_argument(
        "--stop-time",
        type=str,
        default=defaults.get("stop_time", None),
        help=(
            "Optional stop time. Either this or sun elevation condition being met "
            "will end the observing session."
        ),
    )
    limits_group.add_argument(
        "--stop-sun-elevation-deg",
        type=float,
        default=defaults.get("stop_sun_elevation_deg", -14),
        help=(
            "Optional sun elevation threshold. Either this or stop time being met "
            "will end the observing session."
        ),
    )

    return parser.parse_args()


def main():
    """Run the live scheduler with YAML defaults and optional CLI overrides."""

    # parse and validate CLI arguments, with defaults loaded from YAML config
    args = parse_args()
    if args.min_chunk_size is not None:  # XXX implement this
        raise NotImplementedError("min_chunk_size is not implemented yet.")
    if args.start_time is not None:
        args.start_time = time_utils.standardize_time(args.start_time)
    if args.stop_time is not None:
        args.stop_time = time_utils.standardize_time(args.stop_time)

    # initialize requested API client
    print("Initializing blancops Live Scheduler...")
    if args.api_mode.lower() == "mock":
        api = MockTelescopeAPI(exposure_duration=args.mock_exposure_duration)
    else:
        api = BlancoTelescopeAPI()

    # initialize model runner
    if args.model_path_or_alias.lower() == "mock":
        model = MockModelRunner(chunk_size=args.chunk_size)
    else:
        model = AIModelRunner(
            model_path_or_alias=args.model_path_or_alias,
            field_lookup_dir=args.field_lookup_dir,
            fields_path=args.fields_path,
            device=args.device,
            field_choice_method=args.field_choice_method,
            chunk_size=args.chunk_size,
            testing_mode=True,  # XXX check this
        )

    # initialize ui
    if args.ui_mode.lower() == "cli":
        ui = CLIInterface(output_dir=args.output_directory, show_plots=args.show_plots)
    else:
        raise NotImplementedError(f"UI mode '{args.ui_mode}' is not implemented yet.")

    # initialize state manager with session metadata and persisted history
    state = StateManager(
        output_dir=args.output_directory,
        session_id=args.session_id,
        start_time=args.start_time,
        start_sun_elevation=args.start_sun_elevation_deg * units.deg,
        stop_time=args.stop_time,
        stop_sun_elevation=args.stop_sun_elevation_deg * units.deg,
    )

    # initialize orchestrator with all components for running the scheduler loop
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

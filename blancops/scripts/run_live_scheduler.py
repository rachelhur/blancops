import sys
import os

# Ensure blancops is in the path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blancops.live_scheduler.api_client import MockTelescopeAPI, BlancoAPI
from blancops.live_scheduler.model_runner import MockModelRunner, AIModelRunner
from blancops.live_scheduler.interface import CLIInterface
from blancops.live_scheduler.state_manager import StateManager
from blancops.live_scheduler.orchestrator import SchedulerOrchestrator


def main():
    print("Initializing BlancOps Live Scheduler...")

    # parse the YAML config file for the scheduler settings
    # XXX for now, just hardcode the settings directly, but update before deployment
    mode = "mock" # options: "mock", "blanco"
    mock_exposure_duration = 90 # seconds, only used in mock mode
    chunk_size = 3 # number of fields to generate per chunk
    model_path = "default_model.pt" # only used in AI model mode

    # initialize components for running the scheduler
    if mode == "mock":
        api = MockTelescopeAPI(exposure_duration=mock_exposure_duration)
        model = MockModelRunner()
    else:
        api = BlancoTelescopeAPI()
        model = AIModelRunner(model_path=model_path)
    ui = CLIInterface()
    state = StateManager(output_dir="./observing_logs")
    orchestrator = SchedulerOrchestrator(api, model, ui, state, chunk_size=chunk_size)

    # run the scheduler loop
    try:
        orchestrator.run()

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

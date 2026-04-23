"""Main control loop for live scheduling with human approval and telescope polling."""

import time
from blancops.ephemerides import time_utils


class SchedulerOrchestrator:
    """Coordinate API, model, UI, and state manager during a live observing session."""

    def __init__(
        self,
        api,
        model,
        ui,
        state,
        chunk_size=3,
        observing_poll_rate_sec=1,
        telemetry_poll_rate_sec=20,
        # XXX add needed pathing arguments
    ):
        """
        Initialize orchestrator dependencies and loop timing configuration.

        Arguments
        ---------
        api: TelescopeAPI
            Telescope API adapter used for telemetry and queue submission.
        model: ModelRunner
            Model runner that proposes observation chunks.
        ui: BaseInterface
            User interface adapter for review/approval.
        state: StateManager
            State manager handling session history and night boundaries.
        chunk_size: int [3]
            Number of observations generated per proposal chunk.
        observing_poll_rate_sec: float [1]
            Poll cadence while waiting for the current exposure to finish.
        telemetry_poll_rate_sec: float [20]
            Cadence for telemetry checks that can trigger replanning.
        """

        # scheduler components
        self.api = api
        self.model = model
        self.ui = ui
        self.state = state

        # operational settings
        self.chunk_size = chunk_size
        self.observing_poll_rate_sec = observing_poll_rate_sec
        self.telemetry_poll_rate_sec = telemetry_poll_rate_sec

        # track the state of the current session
        self.session_masked_fields = [] # XXX update this logic
        self.first_exposure = True
        self.last_submitted_obs = {}
        self.last_telemetry_check = -float("inf")

    def run(self):
        """Run the continuous proposal/approval/submission loop until end condition."""

        print("\n[Orchestrator] Starting Live Scheduler Loop...")

        # XXX add in pre-loop checks:
        # - get initial telemetry
        # - check initial field lookup
        # - check API connectivity

        while not self.state.check_end_condition():
            # ==========================================================================
            # Generate observing chunk, get user approval
            # ==========================================================================
            telemetry = self.api.get_telemetry()
            # XXX pick up new field files

            # combine manually masked fields with already completed fields
            all_masks = self.session_masked_fields + self.state.completed_fields

            # create the new chunk
            chunk_df = self.model.generate_chunk( # XXX update the call to this function
                telemetry=telemetry,
                available_fields=[],  # XXX Placeholder
                masked_fields=all_masks,
                chunk_size=self.chunk_size,
            )

            # guard against placeholder/failed model output
            if chunk_df is None or chunk_df.empty:
                print("[Orchestrator] Model returned empty chunk. Regenerating...")
                continue

            # get user approval for the chunk before executing
            self.ui.display_chunk(chunk_df)
            approved, new_masks = self.ui.get_user_decision()
            if not approved:
                self.session_masked_fields.extend(new_masks) # XXX we don't want these permanently masked, have it reset after we finally have an approved chunk
                continue

            # ==========================================================================
            # Execute waiting/submission loop with the approved chunk
            # ==========================================================================
            obs_row = chunk_df.iloc[0] # XXX add a loop when we implement min_chunk_size
            exposure_finished = self.api.check_exposure_status()

            # end observing for the night if end condition is met
            if self.state.check_end_condition():
                continue

            # wait for a valid submission point while monitoring interrupts/drift
            submitted = False
            while not submitted:
                # submit the observation the first time through without waiting further
                if self.first_exposure and self.state.check_start_condition():
                    self.api.submit_observation(obs_row)
                    self.last_submitted_obs = obs_row
                    self.first_exposure = False
                    submitted = True
                    print(f"[Orchestrator] First observation submitted: {obs_row['field_id']}")
                    continue

                # otherwise wait for current exposure to finish
                time.sleep(self.observing_poll_rate_sec)
                exposure_finished = self.api.check_exposure_status()

                # submit the next observation if exposure is finished and start condition is met
                if exposure_finished and self.state.check_start_condition():
                    self.api.submit_observation(obs_row)
                    print(
                        f"[Orchestrator] Observation [{obs_row['field_id']}] submitted after [{self.last_submitted_obs['field_id']}] finished."
                    )
                    self.state.record_completion(self.last_submitted_obs)
                    self.last_submitted_obs = obs_row
                    submitted = True
                    continue

                # check for user-triggered soft interrupt to replan chunk
                if self.ui.check_for_replan_signal():
                    print("\n[Orchestrator] User gave soft interrupt. Aborting chunk.")
                    break

                # periodically check for telemetry, field list changes => trigger replan
                delta = time_utils.utc_now() - self.last_telemetry_check
                if delta > self.telemetry_poll_rate_sec:
                    new_telemetry = self.api.get_telemetry()
                    self.last_telemetry_check = time_utils.utc_now()
                    telemetry_changed = False # XXX check telemetry changes
                    if telemetry_changed:
                        print("\n[Orchestrator] Telemetry change detected.")
                    fields_changed = False # XXX check field list changes
                    if fields_changed:
                        print("\n[Orchestrator] Field list change detected.")
                    if telemetry_changed or fields_changed:
                        print(
                            "\n[Orchestrator] Telemetry or field list changed; aborting chunk."
                        )
                        break

        # announce session end
        if self.state.check_end_condition():
            print("[Orchestrator] Observing run complete (end condition met).")
        else:
            print("[Orchestrator] Observing run complete (unknown exit).")

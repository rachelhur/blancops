"""Main control loop for live scheduling with human approval and telescope polling."""

import time
from blancops.ephemerides import time_utils
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class SchedulerOrchestrator:
    """Coordinate client, model, UI, and state manager during a live observing session."""

    def __init__(
        self,
        client,
        model,
        ui,
        progress,
        chunk_size=3,
        min_chunk_size=None,
        observing_poll_rate_sec=1,
        telemetry_poll_rate_sec=20,
        clock=None,
        auto_approve=False,
        # XXX add needed pathing arguments
    ):
        """
        Initialize orchestrator dependencies and loop timing configuration.

        Arguments
        ---------
        client: TelescopeClient
            Telescope client adapter used for telemetry and queue submission.
        model: ModelRunner
            Model runner that proposes observation chunks.
        ui: BaseInterface
            User interface adapter for review/approval.
        progress: ProgressManager
            Progress manager handling session history and night boundaries.
        chunk_size: int [3]
            Number of observations generated per proposal chunk.
        min_chunk_size: int [None]
            Minimum number of un-submitted observations to leave in a chunk before
            replanning; i.e. submit chunk_size - min_chunk_size observations from each
            chunk. Default replans after every submission.
        observing_poll_rate_sec: float [1]
            Poll cadence while waiting for the current exposure to finish.
        telemetry_poll_rate_sec: float [20]
            Cadence for telemetry checks that can trigger replanning.
        clock: Clock, optional
            Clock instance to use for simulated time management. Default to real-time.
        auto_approve: bool [False]
            Whether to skip user approval of proposed chunks. When set, the scheduler
            automatically accepts all proposed chunks.
        """

        # scheduler components
        self.client = client
        self.model = model
        self.ui = ui
        self.progress = progress

        # operational settings
        self.clock = clock or time_utils.Clock()
        self.chunk_size = chunk_size
        self.min_chunk_size = (
            min_chunk_size if min_chunk_size is not None else self.chunk_size - 1
        )
        self.n_to_submit = min(self.chunk_size - self.min_chunk_size, self.chunk_size)
        self.observing_poll_rate_sec = observing_poll_rate_sec
        self.telemetry_poll_rate_sec = telemetry_poll_rate_sec
        self.auto_approve = auto_approve

        # track the state of the current session
        self.session_masked_fields = pd.DataFrame( # XXX Paul: do something with this.....?
            columns=["field_id", "ra", "dec", "filter"]
        )  # field-level operator masks (field_id is the only column AIModelRunner reads)
        # self.session_masked_propids = set()  # propid-level operator masks
        self.masked_field_ids = []
        self.priority_trigger = False
        self.first_exposure = True
        self.last_submitted_obs = {}
        self.last_telemetry_check = -float("inf")

    # def _apply_mask_update(self, to_add, to_remove):
    #     """Apply an operator mask add/remove to the session mask state.

    #     Both arguments are dicts with keys "field_ids" (iterable[int]) and
    #     "propids" (iterable[str]). Field masks are stored in
    #     `session_masked_fields` (field_id populated; ra/dec/filter left NA since
    #     only field_id is consumed downstream); propids in
    #     `session_masked_propids`.
    #     """
    #     add_fids = list(to_add.get("field_ids", []))
    #     if add_fids:
    #         new_rows = pd.DataFrame({"field_id": add_fids}).reindex(
    #             columns=["field_id", "ra", "dec", "filter"]
    #         )
    #         self.session_masked_fields = pd.concat(
    #             [self.session_masked_fields, new_rows], ignore_index=True
    #         ).drop_duplicates(subset=["field_id"], ignore_index=True)
    #     self.session_masked_propids |= set(to_add.get("propids", set()))

    #     remove_fids = set(to_remove.get("field_ids", []))
    #     if remove_fids:
    #         keep = ~self.session_masked_fields["field_id"].isin(remove_fids)
    #         self.session_masked_fields = self.session_masked_fields[keep].reset_index(
    #             drop=True
    #         )
    #     self.session_masked_propids -= set(to_remove.get("propids", set()))

    def run(self):
        """Run the continuous proposal/approval/submission loop until end condition."""

        logger.info("[Orchestrator] Starting Live Scheduler Loop...")

        # XXX add in pre-loop checks:
        # - check initial field lookup
        # - check client connectivity

        # rebuild survey state from this night's log if the session was restarted
        self.model.resume_interrupted_session(self.progress.completed_fields)

        while not self.progress.check_end_condition():
            # ==========================================================================
            # Generate observing chunk, get user approval
            # ==========================================================================
            telemetry = self.client.get_telemetry()
            # XXX pick up new field files

            # create the new chunk
            chunk_df = self.model.generate_chunk(
                telemetry=telemetry,
                available_fields=[],  # XXX Placeholder
                masked_field_ids=self.masked_field_ids,
                priority_trigger=self.priority_trigger,
                chunk_size=self.chunk_size,
            )

            # guard against placeholder/failed model output
            if chunk_df is None or chunk_df.empty:
                logger.warning("[Orchestrator] Model returned empty chunk. Regenerating...")
                continue

            # get user approval for the chunk before executing
            self.ui.display_chunk(
                chunk_df=chunk_df,
                completed_df=self.progress.completed_fields,
                candidate_df=None,
                current=self.last_submitted_obs if len(self.last_submitted_obs) else None,
            )
            if self.auto_approve:
                approved = True
            else:
                approved = self.ui.get_user_decision()
            if not approved: # mask the first field and replan
                to_mask = chunk_df.iloc[0]["field_id"]
                self.masked_field_ids.append(to_mask)
                logger.info(f"[Orchestrator] User rejected the proposed chunk. Masking field: {to_mask}")
                continue
            else: # reset the mask list
                self.masked_field_ids = []

            # ==========================================================================
            # Execute waiting/submission loop with the approved chunk
            # ==========================================================================
            exposure_finished = self.client.check_exposure_status()

            # end observing for the night if end condition is met
            if self.progress.check_end_condition():
                continue

            # wait for a valid submission point while monitoring interrupts/drift
            # submit the first self.n_to_submit observations from the chunk
            logger.info("[Orchestrator] Waiting to execute the approved chunk...")
            submit_idx = 0
            while submit_idx < self.n_to_submit and not self.progress.check_end_condition():
                obs_row = chunk_df.iloc[submit_idx]

                # submit the observation the first time through without waiting further
                if self.first_exposure and self.progress.check_start_condition():
                    self.client.submit_observation(obs_row)
                    self.model.record_visit(obs_row)
                    self.last_submitted_obs = obs_row
                    self.first_exposure = False
                    logger.info(
                        f"[Orchestrator] First observation submitted: {obs_row['field_id']}"
                    )
                    submit_idx += 1
                    continue # move to next observation in the chunk

                # otherwise wait for current exposure to finish
                time.sleep(self.observing_poll_rate_sec)
                exposure_finished = self.client.check_exposure_status()

                # submit the next observation if exposure is finished and start condition is met
                if (
                    not self.first_exposure
                    and exposure_finished
                    and self.progress.check_start_condition()
                ):
                    self.client.submit_observation(obs_row)
                    self.progress.record_completion(obs_row)
                    self.model.record_visit(obs_row)
                    logger.info(
                        f"[Orchestrator] Observation {obs_row['field_id']} submitted after {self.last_submitted_obs['field_id']} finished."
                    )
                    #self.progress.record_completion(self.last_submitted_obs)
                    self.last_submitted_obs = obs_row
                    submit_idx += 1
                    continue # move to next observation in the chunk

                # check for user-triggered soft interrupt to replan chunk
                replan = self.ui.check_for_replan_signal()
                if replan[0] and replan[1] == "replan":
                    self.masked_field_ids.append(obs_row["field_id"])
                    logger.info(
                        f"[Orchestrator] User requested a replan. Aborting chunk and masking field: {obs_row['field_id']}"
                    )
                    break # exit the submission loop to replan the chunk

                # check for user-triggered gravitational wave trigger
                if replan[0] and replan[1] == "gw-trigger":
                    logger.info("[Orchestrator] User triggered gravitational-wave follow-up observations.")
                    self.priority_trigger = True
                    break # exit the submission loop to replan the chunk

                # periodically check for telemetry, field list changes => trigger replan
                delta = self.clock.now() - self.last_telemetry_check
                if delta > self.telemetry_poll_rate_sec:
                    logger.info("[Orchestrator] Performing periodic telemetry/field check")
                    new_telemetry = self.client.get_telemetry()
                    self.last_telemetry_check = self.clock.now()
                    telemetry_changed = self.client.check_telemetry_change()
                    if telemetry_changed:
                        logger.info("[Orchestrator] Telemetry change detected.")
                    fields_changed = False  # XXX check field list changes
                    if fields_changed:
                        logger.info("[Orchestrator] Field list change detected.")
                    if telemetry_changed or fields_changed:
                        logger.info(
                            "[Orchestrator] Telemetry or field list changed; aborting chunk."
                        )
                        break # exit the submission loop to replan the chunk

        # record the last submitted observation
        # NOTE self.last_submitted_obs is a pd.Series, need
        # to either convert to dict or use len()
        #if len(self.last_submitted_obs):
        #    self.progress.record_completion(self.last_submitted_obs)

        # announce session end
        if self.progress.check_end_condition():
            logger.info("[Orchestrator] Observing run complete (end condition met).")
        else:
            logger.info("[Orchestrator] Observing run complete (unknown exit).")
        self.client.close()

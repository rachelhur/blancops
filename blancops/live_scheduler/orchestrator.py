import time

class SchedulerOrchestrator:
    def __init__(self, api, model, ui, state, chunk_size=3):
        self.api = api
        self.model = model
        self.ui = ui
        self.state = state
        self.chunk_size = chunk_size
        self.session_masked_fields = [] # Fields manually masked by the user tonight
        self.first_exposure = True
        self.last_submitted_obs = []
        self.last_telemetry_check = -float("inf")

    def run(self):
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
            
            # Combine manually masked fields with already completed fields
            all_masks = self.session_masked_fields + self.state.completed_fields
            
            # create the new chunk
            chunk_df = self.model.generate_chunk(
                telemetry=telemetry, 
                available_fields=[], # Placeholder
                masked_fields=all_masks,
                chunk_size=self.chunk_size
            )

            self.ui.display_chunk(chunk_df)
            approved, new_masks = self.ui.get_user_decision()

            if not approved:
                self.session_masked_fields.extend(new_masks)
                continue

            # ==========================================================================
            # Execute waiting/submission loop with the approved chunk
            # ==========================================================================
            obs_row = chunk_df.iloc[0]
            exposure_finished = self.api.check_exposure_status()

            # end observing for the night if end condition is met
            if self.state.check_end_condition():
                print("[Orchestrator] End condition met. Shutting down.")
                return

            # the waiting loop
            submitted = False
            while not submitted:

                # submit the observation the first time through without waiting further
                if self.first_exposure and self.state.check_start_condition():
                    self.api.submit_observation(obs_row)
                    self.last_submitted_obs = obs_row
                    self.first_exposure = False
                    submitted = True
                    print("[Orchestrator] First observation submitted.")
                    continue

                # otherwise wait for current exposure to finish
                time.sleep(1) # XXX this amount should be an argument
                exposure_finished = self.api.check_exposure_status()

                # submit the next observation if exposure is finished and start condition is met
                if exposure_finished and self.state.check_start_condition():
                    self.state.record_completion(self.last_submitted_obs)
                    self.api.submit_observation(obs_row)
                    self.last_submitted_obs = obs_row
                    submitted = True
                    print("[Orchestrator] Observation submitted after exposure finished.")
                    continue

                # check for user-triggered soft interrupt to replan chunk
                if self.ui.check_for_replan_signal():
                    print("\n[Orchestrator] User triggered soft interrupt. Aborting chunk.")
                    break

                # check telemetry periodically to see if we need to replan the chunk due to changes in conditions or field visibility
                if time.time() - self.last_telemetry_check > 20: # XXX this amount should be an argument
                    self.last_telemetry_check = time.time()
                    new_telemetry = self.api.get_telemetry()
                    telemetry_changed = False # XXX placeholder: check telemetry changes
                    fields_changed = False # XXX placeholder: check telemetry changes
                    if telemetry_changed or fields_changed:
                        print("\n[Orchestrator] Telemetry or field actions changed. Aborting chunk for replan.")
                        break

            # If loop broke due to interrupt or drift, break out of chunk execution
            if not submitted:
                print("[Orchestrator] Wiping remaining unapproved chunk.")
                break 
            
                
        print("Observing night complete. Sun elevation limit reached.")
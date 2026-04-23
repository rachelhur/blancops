import json
import os
from datetime import datetime, timedelta


class StateManager:
    def __init__(self, output_dir, session_id=None):
        self.output_dir = output_dir
        self.session_id = session_id or self._generate_session_id()

        self.history_file = os.path.join(
            self.output_dir, f"observing_log_{self.session_id}.jsonl"
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.completed_fields = self._load_history()

    # XXX add method for logging all print statements to a file in addition to terminal

    def _generate_session_id(self):
        """Maps continuous observing blocks across midnight to a single date."""
        now = datetime.now()
        # If before noon, it belongs to the previous day's observing night
        if now.hour < 12:
            session_date = now - timedelta(days=1)
        else:
            session_date = now
        session_date = session_date.strftime("%Y-%m-%d")
        print(f"[State] Generated session ID: {session_date}")
        return session_date

    def _load_history(self):
        fields = []
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                for line in f:
                    fields.append(json.loads(line)["field_id"])
            print(
                f"[State] Resumed with {len(fields)} completed fields from {self.history_file}."
            )
        return fields

    def record_completion(self, obs_row):
        obs_dict = obs_row.to_dict()
        with open(self.history_file, "a") as f:
            f.write(json.dumps(obs_dict) + "\n")
        self.completed_fields.append(obs_dict.get("field_id"))

    def check_start_condition(self):
        # XXX call start condition checks here
        return True

    def check_end_condition(self):
        # XXX call end condition checks here
        return False

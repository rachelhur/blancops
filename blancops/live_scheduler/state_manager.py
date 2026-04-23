"""Persistent session state and observing-history management for the live scheduler."""

import json
import os
from datetime import datetime, timedelta


class StateManager:
    """Track the current observing session and log completed fields to disk."""

    def __init__(self, output_dir, session_id=None):
        """
        Initialize session metadata, output paths, and persisted history.

        Arguments
        ---------
        output_dir: str
            Directory where observing history should be written.
        session_id: str, optional
            Explicit observing-session identifier. If omitted, one is generated.
        """

        self.output_dir = output_dir
        self.session_id = session_id or self._generate_session_id()
        self.history_file = os.path.join(
            self.output_dir, f"observing_log_{self.session_id}.jsonl"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.completed_fields = self._load_history()

    # XXX add a log sink so important scheduler messages are also written to disk

    def _generate_session_id(self):
        """Map a continuous observing block across midnight to a single session date."""

        # map the current time to a start date; if before noon use the previous day
        now = datetime.now()
        if now.hour < 12:
            session_date = now - timedelta(days=1)
        else:
            session_date = now
        session_date = session_date.strftime("%Y-%m-%d")
        print(f"[State] Generated session ID: {session_date}")
        return session_date

    def _load_history(self):
        """Load completed field IDs from the existing JSONL history file."""

        # resume by replaying the existing JSONL history file line by line
        fields = []
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                for line in f:
                    line = line.strip()

                    # ignore blank lines so a partial write does not break recovery
                    if not line:
                        continue

                    # skip malformed lines instead of failing the whole restart
                    try:
                        field_id = json.loads(line).get("field_id")
                    except json.JSONDecodeError:
                        print(
                            f"[State] Skipping malformed history line in {self.history_file}."
                        )
                        continue

                    # store the successfully parsed field ID
                    if field_id is not None:
                        fields.append(field_id)

            print(
                f"[State] Resumed with {len(fields)} completed fields from {self.history_file}."
            )
        return fields

    def record_completion(self, obs_row):
        """
        Append a completed observation to the session history.

        Arguments
        ---------
        obs_row: dict or pandas.Series
            Completed observation containing at least a field_id
        """
        # store each JSON object sequentially on a new line as they are completed
        obs_dict = obs_row.to_dict() if hasattr(obs_row, "to_dict") else dict(obs_row)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(obs_dict) + "\n")
        field_id = obs_dict.get("field_id")
        if field_id is not None:
            self.completed_fields.append(field_id)

    def check_start_condition(self):
        """
        Return whether conditions have been met to start the observing session.
        """

        # XXX call start condition checks here.
        return True

    def check_end_condition(self):
        """
        Return whether conditions have been met to end the observing session.
        """

        # XXX call end condition checks here.
        return False

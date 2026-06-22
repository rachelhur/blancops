"""User-interface adapters for approving scheduler observation chunks.

This module defines the scheduler-facing UI contract and a CLI implementation used for
local, human-in-the-loop operation.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from blancops.math import units
from blancops.plotting import live_scheduling_viz
from blancops.ephemerides import time_utils
import sys, select
import pathlib
import logging
logger = logging.getLogger(__name__)

class BaseInterface(ABC):
    """Abstract interface for user interaction with proposed observation chunks."""

    @abstractmethod
    def __init__(self, output_dir, show_plots):
        """
        Initialize the interface.

        Arguments
        ---------
        output_dir: str or pathlib.Path, optional
            Directory to save any generated outputs (e.g. plots).
        show_plots: bool
            Whether to display plots interactively or just save to disk.
        """

        pass

    @abstractmethod
    def display_chunk(self, chunk_df):
        """
        Render the proposed observation chunk for operator review.

        Arguments
        ---------
        chunk_df: pandas.DataFrame
            Proposed observations and associated metadata.
        """

        pass

    @abstractmethod
    def get_user_decision(self):
        """
        Get user approval/rejection for the current proposed chunk.

        Returns
        -------
        approved: bool
            Whether the user approves the proposed chunk for execution.
        masked_fields: list
            List of field IDs to mask in the next proposal if the chunk is rejected.
        """

        pass

    @abstractmethod
    def check_for_replan_signal(self):
        """
        Check whether operator requested an asynchronous chunk replan.

        Returns
        -------
        bool
            True if a replan should be triggered.
        """

        pass


class CLIInterface(BaseInterface):
    """Command-line interface for chunk preview and approval."""

    def __init__(self, output_dir=None, show_plots=True, clock=None):
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.show_plots = show_plots
        self.clock = clock or time_utils.Clock()
        if self.output_dir is None and not self.show_plots:
            logger.warning("[Interface] Warning: No plots will be saved or displayed.")

    def display_chunk(self, chunk_df):
        """Print the proposed chunk and save a simple RA/Dec plot."""

        # print the proposed chunk as a table in the terminal
        logger.info(
                    "\n" + "=" * 88
                    + "\n Proposed Observing Chunk \n"
                    + "=" * 88
                    + "\n" + chunk_df.to_string(index=False)
                    + "\n" + "=" * 88
                    )

        # skip plotting when upstream returns an empty/malformed proposal
        required_cols = {"ra", "dec"}
        if chunk_df.empty:
            logger.info("[Interface] Chunk is empty; skipping plot generation.")
            return
        if not required_cols.issubset(chunk_df.columns):
            logger.info(
                "[Interface] Chunk missing required ra/dec columns; skipping plot generation."
            )
            return

        # Generate and save a plot for quick visual inspection.
        # XXX update to include completed, future, and current fields in the plot
        # center and time-stamp the plot on the scheduler clock so the simulated
        # time is respected during testing and the true UTC during live runs
        live_scheduling_viz.plot_live_schedule_snapshot(
            proposed_df=chunk_df, time=self.clock.now()
        )
        if self.output_dir is not None:
            plt.savefig(self.output_dir / "current_chunk_proposal.png")
            logger.info(
                f"[Interface] Plot saved to '{self.output_dir / 'current_chunk_proposal.png'}'."
            )
        if self.show_plots:
            plt.show(block=False)
            plt.pause(0.1)

    def get_user_decision(self):
        """Prompt for Y/N approval and return scheduler decision payload."""

        while True:
            resp = (
                input("Approve this chunk? [Y]es, [N]o (mask fields): ").strip().upper()
            )
            if resp == "Y":
                logger.info("[Interface] Chunk accepted.")
                return True
            elif resp == "N":
                logger.info("[Interface] Chunk rejected.")
                return False
            else:
                logger.warning("[Interface] Invalid input, please enter Y or N.")

    def check_for_replan_signal(self):
        """
        Non-blocking soft-interrupt check using standard select. Performs the following
        user-input checks:
        - r + Enter: signal to replan the current chunk
        - t + Enter: (temporary) signal gravitational-wave follow-up observations

        Returns
        -------
        bool, str
            Tuple of (signal_received, signal_type) where signal_received indicates
            whether a valid signal was detected, and signal_type is a string label for
            the type of signal (e.g. 'replan', 'gw_trigger') or None
        """
        # select.select checks if a file descriptor is ready to be read
        # [sys.stdin] is the list of objects we want to check
        # 0.0 timeout: tell select to return instantly (non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
            
            # entering this block means user hit Enter; read the line and clean buffer
            user_input = sys.stdin.readline().strip().lower()
            if user_input == 'r':
                logger.info("[Interface] Soft interrupt detected! 'r' entered.")
                return True, 'replan'
            elif user_input == 't':
                logger.info("[Interface] Gravitational-wave trigger signal detected! 't' entered.")
                return True, 'gw-trigger'
            else: # ignore other input; note that reading the input cleared the buffer
                pass
                
        return False, None

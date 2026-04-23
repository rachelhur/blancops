"""User-interface adapters for approving scheduler observation chunks.

This module defines the scheduler-facing UI contract and a CLI implementation
used for local, human-in-the-loop operation.
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from blancops.math import units


class BaseInterface(ABC):
    """Abstract interface for user interaction with proposed observation chunks."""

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

    def display_chunk(self, chunk_df):
        """Print the proposed chunk and save a simple RA/Dec plot."""

        # print the proposed chunk as a table in the terminal
        print("\n" + "=" * 88)
        print("[Interface] Proposed Observing Chunk")
        print("=" * 88)
        print(chunk_df.to_string(index=False))
        print("=" * 88)

        # skip plotting when upstream returns an empty/malformed proposal
        required_cols = {"ra", "dec"}
        if chunk_df.empty:
            print("[Interface] Chunk is empty; skipping plot generation.")
            return
        if not required_cols.issubset(chunk_df.columns):
            print(
                "[Interface] Chunk missing required ra/dec columns; skipping plot generation."
            )
            return

        # Generate and save a plot for quick visual inspection.
        # XXX upgrade to a sky plot from the plotting utilities
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            chunk_df["ra"] / units.degree,
            chunk_df["dec"] / units.degree,
            c=range(len(chunk_df)),
            cmap="viridis",
        )
        plt.colorbar(sc, label="Observation Order")
        plt.xlabel("RA [deg]")
        plt.ylabel("Dec [deg]")
        plt.title("Proposed Chunk Pointings")
        plt.grid(True)
        plt.savefig("current_chunk_proposal.png")
        plt.close()
        print("[Interface] Plot saved to 'current_chunk_proposal.png'.")

    def get_user_decision(self):
        """Prompt for Y/N approval and return scheduler decision payload."""

        while True:
            resp = (
                input("Approve this chunk? [Y]es, [N]o (mask fields): ").strip().upper()
            )
            if resp == "Y":
                print("[Interface] Chunk accepted.")
                return True
            elif resp == "N":
                print("[Interface] Chunk rejected.")
                return False
            else:
                print("[Interface] Invalid input, please enter Y or N.")

    def check_for_replan_signal(self):
        """CLI has no non-blocking soft-interrupt channel; always returns False."""

        return False

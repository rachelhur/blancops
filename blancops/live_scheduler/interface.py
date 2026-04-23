from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from blancops.math import units


# Abstract Base Class for User Interfaces (CLI, Web UI, etc.)
class BaseInterface(ABC):
    # display the proposed observing chunk to the user (as a DataFrame, plot, etc.)
    @abstractmethod
    def display_chunk(self, chunk_df):
        pass

    # after generating a chunk plan, get user approval or rejection (regenerate plan)
    @abstractmethod
    def get_user_decision(self):
        pass

    # soft interrupt, allowing a user to change their mind and signal a chunk replan
    @abstractmethod
    def check_for_replan_signal(self):
        pass


# CLI Implementation of the User Interface
class CLIInterface(BaseInterface):
    def display_chunk(self, chunk_df):
        # print the proposed chunk as a table in the terminal
        print("\n" + "=" * 88)
        print("[Interface] Proposed Observing Chunk")
        print("=" * 88)
        print(chunk_df.to_string(index=False))
        print("=" * 88)

        # generate and save a plot of the chunk for visual inspection
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
        while True:
            resp = (
                input("Approve this chunk? [Y]es, [N]o (mask fields): ").strip().upper()
            )
            if resp == "Y":
                print("[Interface] Chunk accepted. Waiting to submit observation...")
                return True, []
            elif resp == "N":
                print("[Interface] Chunk rejected. Generating a new plan...")
                # Placeholder: In a real CLI, we might ask for specific IDs to mask here
                return False, [101]
            else:
                print("Invalid input, please enter Y or N.")

    def check_for_replan_signal(self):
        # Soft Interrupt always False for CLI, as keyboard input blocks background loops
        return False

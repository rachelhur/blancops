import os
import argparse
from pathlib import Path
# import importlib.resources as pkg_resources
from importlib import resources

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2]

def main():
    """
    Initialized a workspace for blancops and saves a pointer to ~/.blancops_profile. Default workspace is the project root directory, blancops
    """
    parser = argparse.ArgumentParser(description="Initialize blancops workspace and saves a pointer to ~/.blancops_profile")
    parser.add_argument(
        '--workspace',
        '-w',
        type=Path, 
        default=Path(os.getenv("BLANCOPS_WORKSPACE", PROJECT_DIR)),
        help="Target directory to initialize the workspace. Defaults to project root, `blancops`"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Overwrite existing configuration files if they already exist."
    )
    
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    logger.info(f"Initializing workspace at: {workspace}")

    # Create the necessary directory structure
    directories_to_create = [
        workspace,
        workspace / "configs",
        workspace / "experiments",
        workspace / "deployable_models",
        workspace / "data" / "train",
        workspace / "data" / "test_suite"
    ]
    
    for dir_path in directories_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  [+] Created directory: {dir_path}")

    # save workspace pointer file
    pointer_file = Path.home() / ".blancops_profile"
    pointer_file.write_text(str(workspace))
    logger.info(f"  [+] Saved workspace pointer to {pointer_file}")

    logger.info("\nInitialization complete!")

if __name__ == "__main__":
    main()
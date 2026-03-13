import os
import argparse
from pathlib import Path
# import importlib.resources as pkg_resources
from importlib import resources
from blancops.data_processing.data_processing import save_DES_bin_and_field_mappings

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2]

def main():
    """
    .
    Default workspace is the project root directory, blancops
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

    # 1. Create the necessary directory structure
    directories_to_create = [
        workspace,
        workspace / "configs",
        workspace / "experiments",
        workspace / "models",
        workspace / "data" / "lookups"
    ]
    
    for dir_path in directories_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  [+] Created directory: {dir_path}")

    # 2. Copy the default global_config.json out of the package
    config_dict = {
        "global_config.json": workspace / "configs" / "global_config.json",
        "template_train_config.json": workspace / "configs" / "template_train_config.json"
    }
    
    for cfg_name, cfg_dest in config_dict.items():
        if cfg_dest.exists() and not args.force:
            logger.warning(f" [!] Config already exists at {cfg_dest}. Use --force to overwrite.")
        else:
            try:
                # Copy global_config.json from within package to config_dest
                config_text = resources.files('blancops.configs').joinpath(cfg_name).read_text()
                cfg_dest.write_text(config_text)
                logger.info(f"  [+] Copied default {cfg_name} to: {cfg_dest}")
            except Exception as e:
                logger.warning(f"  [!] Failed to copy config. Reason: {e}")

    try:
        save_DES_bin_and_field_mappings(fits_path= workspace / "data" / "fits" / "decam-exposures-20251211.fits", outdir=workspace / "data" / "lookups")
        logger.info(f" [!] Constructed train data lookup tables")
    except Exception as e:
        logger.warning(f" [!] Failed to construct train data lookup tables. Reason: {e}")

    # save workspace pointer file
    pointer_file = Path.home() / ".blancops_profile"
    pointer_file.write_text(str(workspace))
    logger.warning(f"  [+] Saved workspace pointer to {pointer_file}")

    logger.warning("\nInitialization complete!")

if __name__ == "__main__":
    main()
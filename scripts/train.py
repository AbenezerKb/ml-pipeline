import sys
from pathlib import Path
import logging
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.pre_processing import DataPreparation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    logger.info("Starting ML Training Pipeline")
    
    try:
        logger.info("\nData Preparation")
        data_prep = DataPreparation(config_path)
        train_df, test_df = data_prep.run()
        
       
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yml",
        help="Path to training config file"
    )
    
    args = parser.parse_args()
    main(args.config)




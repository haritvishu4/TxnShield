import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.data.ingestion import DataIngestion
from src.utils.logger import get_logger

logger = get_logger("script_download_data")

def main():
    parser = argparse.ArgumentParser(description="Download credit card fraud dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download if file already exists")
    args = parser.parse_args()

    logger.info("Starting dataset acquisition...")
    ingestion = DataIngestion()
    ingestion.download_dataset(force=args.force)
    df = ingestion.load_data()
    summary = ingestion.validate_dataset_integrity(df)
    logger.info("Dataset acquisition completed successfully.")

if __name__ == "__main__":
    main()

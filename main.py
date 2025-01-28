import logging
from scripts import (
    data_ingestion,
    data_preprocessing,
    model_training,
    model_evaluation
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting the pipeline...")

    logging.info("Running data ingestion...")
    data_ingestion.main()

    logging.info("Running data preprocessing...")
    data_preprocessing.main()

    logging.info("Running model training...")
    model_training.main()

    logging.info("Running model evaluation...")
    model_evaluation.main()

    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

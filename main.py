import logging
from scripts import (
    data_ingestion,
    data_preprocessing,
    model_loader,
    model_training,
    training_UNETR,
    training_VNET,
    model_evaluation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("Starting the pipeline...")

    logging.info("Running data_ingestion.py...")
    data_ingestion.main()

    logging.info("Running data_preprocessing.py...")
    data_preprocessing.main()

    logging.info("Running training_UNETR.py...")
    model_training.main()

    logging.info("Running model_evaluation_mc.py...")
    model_evaluation.main()

    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()

from enum import Enum
from pathlib import Path


class ModelConfig(Enum):

    EPOCHS = 30
    BATCH_SIZE = 64
    LR = 1e-3

    LOG_DIR = Path("loggs")

    RUNs_LOG_DIR = f"{LOG_DIR}/runs"
    LOG_CHECKPOINTS_DIR = f"{LOG_DIR}/checkpoints"

    LOG_METRICSS_DIR = f"{LOG_DIR}/metrics"
    LOG_CF_MATRIX = f"{LOG_METRICSS_DIR}/confusion_matrix_epoch_"
    LOG_CLS_REPORT = f"{LOG_METRICSS_DIR}/classification_report_epoch_"
    LOG_F1 = f"{LOG_METRICSS_DIR}/f1_scores.txt"


if __name__ == "__main__":
    print(ModelConfig.EPOCHS.value)
    print(ModelConfig.BATCH_SIZE.value)
    print(ModelConfig.LR.value)

# python -m src.enums.ModelEnums

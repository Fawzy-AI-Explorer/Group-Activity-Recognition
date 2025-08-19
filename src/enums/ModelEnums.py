from enum import Enum
from pathlib import Path


class ModelConfig(Enum):

    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3

    LOG_DIR = Path("loggs")
    RUNs_LOG_DIR = f"{LOG_DIR}/runs"

if __name__ == "__main__":
    print(ModelConfig.EPOCHS.value)
    print(ModelConfig.BATCH_SIZE.value)
    print(ModelConfig.LR.value)

# python -m src.enums.ModelEnums

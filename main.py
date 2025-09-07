import os

# Limit threading BEFORE TensorFlow is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disable oneDNN threads (optional but safer on HPC)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from data_loader import VideoDataset
from model import cusModel, PreModel
from trainer import Trainer
from config import CFG

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU")
    except RuntimeError as e:
        print(e)


def main():
    dataset = VideoDataset()
    train_ds, valid_ds = dataset.get_datasets()

    if CFG.model_select == "custom_model":
        print("Using custom 3D CNN model")
        model = cusModel.build()
        model.summary()

    elif CFG.model_select == "pre_model":
        print("Using Pre-trained model")
        model = PreModel.build()
        # Build model by running a dummy batch
        dummy_input = tf.random.normal((1, 10, 224, 224, 3))
        _ = model(dummy_input)
        model.summary()

    else:
        raise ValueError(f"Unknown model type: {CFG.model_select}")

    trainer = Trainer(model, train_ds, valid_ds)
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()

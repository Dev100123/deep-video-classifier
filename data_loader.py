import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from config import CFG
from utils import frames_from_video_file

class VideoDataset:
    def __init__(self):
        self.file_paths = []
        self.targets = []

    def load_files(self):
        for i, cls in enumerate(CFG.classes):
            sub_file_paths = glob.glob(f"{CFG.dataset_path}/{cls}/**.mp4")
            self.file_paths += sub_file_paths
            self.targets += [i] * len(sub_file_paths)

    def extract_features(self):
        features = []
        for file_path in tqdm(self.file_paths):
            features.append(frames_from_video_file(file_path, CFG.n_frames, CFG.output_size, CFG.frame_step))
        return np.array(features)

    def get_datasets(self):
        self.load_files()
        features = self.extract_features()

        train_features, val_features, train_targets, val_targets = train_test_split(
            features, self.targets, test_size=CFG.test_size, random_state=CFG.random_state
        )
        
        print(f"Train samples: {len(train_targets)} | Validation samples: {len(val_targets)}")

        train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)) \
            .shuffle(CFG.batch_size * 4).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        valid_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)) \
            .batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)

        del train_features, val_features
        gc.collect()

        return train_ds, valid_ds

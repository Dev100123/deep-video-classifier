# ================================
# HARD ENV LIMITS (MUST BE FIRST)
# ================================
import os

# ---- GPU control ----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set "-1" for CPU-only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUMA_DISABLED"] = "1"

# ---- HARD THREAD LIMITS ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"

# ================================
# IMPORTS (AFTER ENV VARS)
# ================================
import tensorflow as tf
import cv2
import numpy as np
import random
import pandas as pd
import gc

# ================================
# DISABLE OPENCV THREADS (CRITICAL)
# ================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ================================
# TENSORFLOW THREAD CONTROL
# ================================
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Disable tf.data background threads
tf.data.experimental.enable_debug_mode()

# ================================
# CONFIG
# ================================
class CFG:
    n_frames = 10
    output_size = (224, 224)   # reduce to (160,160) if memory issues
    frame_step = 15
    classes = ["orginal", "watermarked"]

# ================================
# FRAME PREPROCESSING
# ================================
def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

# ================================
# VIDEO → FRAMES
# ================================
def frames_from_video_file(
    video_path,
    n_frames=10,
    output_size=(224, 224),
    frame_step=15
):
    src = cv2.VideoCapture(str(video_path))
    if not src.isOpened():
        return None

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length = 1 + (n_frames - 1) * frame_step

    start = 0 if need_length > video_length else random.randint(
        0, max(0, video_length - need_length)
    )
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    ret, frame = src.read()
    if not ret:
        src.release()
        return None

    frames.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
            if not ret:
                break
        if ret:
            frames.append(format_frames(frame, output_size))
        else:
            frames.append(tf.zeros_like(frames[0]))

    src.release()

    frames = tf.stack(frames)

    # ✅ TensorFlow-safe BGR → RGB
    frames = tf.reverse(frames, axis=[-1])

    return frames.numpy()


# ================================
# 3D CNN MODEL
# ================================
def build_model(input_shape=(10, 224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((1, 2, 2))(x)

    x = tf.keras.layers.Conv3D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

# ================================
# SINGLE VIDEO PREDICTION
# ================================
def predict_video_class(video_path, model):
    frames = frames_from_video_file(
        video_path,
        n_frames=CFG.n_frames,
        output_size=CFG.output_size,
        frame_step=CFG.frame_step
    )

    if frames is None:
        return None, None

    frames = np.expand_dims(frames, axis=0)
    prob = model(frames, training=False).numpy()[0][0]
    pred_class = CFG.classes[int(prob > 0.5)]

    return pred_class, float(prob)

# ================================
# FOLDER INFERENCE
# ================================
def process_videos_in_folder(folder_path, model, output_csv):
    results = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")

                try:
                    pred_class, prob = predict_video_class(video_path, model)
                    if pred_class is not None:
                        results.append({
                            "video_name": file,
                            "predicted_class": pred_class
                        })
                except Exception as e:
                    print(f"⚠️ Failed: {file} ({e})")

                gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    model = build_model()
    model.load_weights(
        "/lustre/home/rchandraghosh/deep-video-classifier-main/"
        "kfold_results/best_model_fold_2.h5"
    )

    print("✅ Binary 3D CNN model loaded successfully!")

    folder_path = (
        "/ibiscostorage/rchandraghosh/video_dataset/"
        "testing_dataset/6_temporal_edits/watermarked"
    )

    process_videos_in_folder(
        folder_path,
        model,
        output_csv="temporal_edits_watermarked.csv"
    )

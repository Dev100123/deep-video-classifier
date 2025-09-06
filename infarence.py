import os
import cv2
import numpy as np
import tensorflow as tf
import random
import pandas as pd

# --- Configuration ---
class CFG:
    n_frames = 10
    output_size = (224, 224)
    frame_step = 15
    classes = ["orginal", "watermark"]  # Replace with your actual class names

# --- Frame formatting function ---
def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

# --- Extract frames from video ---
def frames_from_video_file(video_path, n_frames=10, output_size=(224,224), frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step
    start = 0 if need_length > video_length else random.randint(0, int(video_length - need_length) + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if not ret:
        src.release()
        return None  # invalid video
    
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]  # BGR → RGB
    return result

# --- Build the same 3D CNN model ---
# def build_model(input_shape=(10, 224, 224, 3)):
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=input_shape),
#         tf.keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu"),
#         tf.keras.layers.MaxPooling3D(),
#         tf.keras.layers.Conv3D(64, kernel_size=3, padding="same", activation="relu"),
#         tf.keras.layers.MaxPooling3D(),
#         tf.keras.layers.Conv3D(128, kernel_size=3, padding="same", activation="relu"),
#         tf.keras.layers.MaxPooling3D(),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.GlobalAveragePooling3D(),
#         tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
#     ])
#     return model

# --- Build the pre-trained model ---
def build_model():
    net = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet"
    )
    net.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(255.0),
        tf.keras.layers.TimeDistributed(net),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        tf.keras.layers.GlobalAveragePooling3D(), 
    ])
    return model

# --- Prediction function for binary model ---
def predict_video_class(video_path, model, n_frames=10, output_size=(224,224), frame_step=15):
    frames = frames_from_video_file(video_path, n_frames=n_frames, output_size=output_size, frame_step=frame_step)
    if frames is None:
        return None, None
    frames = np.expand_dims(frames, axis=0)  # add batch dimension
    pred_prob = model.predict(frames, verbose=0)[0][0]  # probability of class 1
    pred_class = CFG.classes[int(pred_prob > 0.5)]
    return pred_class, float(pred_prob)

# --- Process all videos in a folder ---
def process_videos_in_folder(folder_path, model, output_csv="results.csv"):
    results = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):  # valid video formats
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")
                
                pred_class, pred_prob = predict_video_class(
                    video_path, model,
                    n_frames=CFG.n_frames,
                    output_size=CFG.output_size,
                    frame_step=CFG.frame_step
                )
                
                if pred_class is not None:
                    results.append({
                        "video_name": file,
                        "predicted_class": pred_class,
                        #"probability": pred_prob
                    })
                else:
                    print(f"⚠️ Skipped unreadable video: {file}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")

# --- Main inference ---
if __name__ == "__main__":
    # Build model and load weights
    model = build_model()
    dummy_input = np.zeros((1, 10, 224, 224, 3))
    model(dummy_input)
    model.load_weights(r"C:\Users\RAJIB\Desktop\deep-video-classifier-main\saved data\no_noice_original_vgg16_eps_0.05\model.h5")
    print("Binary 3D CNN model loaded successfully!")

    # Path to your video dataset folder
    #folder_path = r"C:\Users\RAJIB\Desktop\watermark_video_blockchain\dataset_video\testing\noisy_videos_original-20250820T105017Z-1-001\noisy_videos_original\new"
    #folder_path = r"C:\Users\RAJIB\Desktop\watermark_video_blockchain\dataset_video\testing\noisy_videos_watermarked-20250820T104128Z-1-001\noisy_videos_watermarked\new"
    #folder_path = r"C:\Users\RAJIB\Desktop\watermark_video_blockchain\dataset_video\testing\test_original_videos-20250820T103545Z-1-001\test_original_videos"
    folder_path = r"C:\Users\RAJIB\Desktop\watermark_video_blockchain\dataset_video\testing\watermarked-20250820T102841Z-1-001"

    # Run predictions on all videos & save results
    #process_videos_in_folder(folder_path, model, output_csv="noise_video_original_result.csv")
    #process_videos_in_folder(folder_path, model, output_csv="noise_video_watermark_result.csv")
    #process_videos_in_folder(folder_path, model, output_csv="original_result.csv")
    process_videos_in_folder(folder_path, model, output_csv="watermark_result.csv")

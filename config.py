class CFG:
    classes = ["original", "watermark"] 
    dataset_path = r"C:\Users\RAJIB\Desktop\watermark_video_blockchain\dataset\training"
    batch_size = 32
    n_frames = 10
    output_size = (224, 224)
    frame_step = 15
    test_size = 0.2
    random_state = 42
    epochs =3
    checkpoint_path = "model.h5"

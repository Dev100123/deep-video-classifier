class CFG:
    classes = ["original", "watermark"] 
    dataset_path = r"/ibiscostorage/rchandraghosh/video_dataset/dataset with noice eps 0.05/training"
    batch_size = 32
    n_frames = 10
    output_size = (224, 224)
    frame_step = 15
    test_size = 0.2
    random_state = 42
    epochs = 30
    model_select = "pre_model"  # options: pre_model/custom_model
    model_name = "resnet50"  # options: vgg16, vgg19, resnet50
    checkpoint_path = "model.h5"

from data_loader import VideoDataset
from model_builder import VideoModel
from trainer import Trainer

#if you copy my code dont forget to thank us😁
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

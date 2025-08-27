from data_loader import VideoDataset
from model_builder import VideoModel
from trainer import Trainer

#if you copy my code dont forget to thank us😁
def main():
    dataset = VideoDataset()
    train_ds, valid_ds = dataset.get_datasets()

    model = VideoModel.build()
    model.summary()

    trainer = Trainer(model, train_ds, valid_ds)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main() #try to do something good always✌️

from src.training import download_dataset,train_model

if __name__ == "__main__":
    dataset_path = download_dataset()
    train_model(dataset_path)
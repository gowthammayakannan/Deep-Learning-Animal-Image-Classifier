from src.data_loader import extract_zip, load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_metrics
from src.predict import predict_random

ZIP_PATH = "data/Animals.zip"
EXTRACT_PATH = "data/Animals"

if __name__ == "__main__":

    # 1. Extract dataset
    extract_zip(ZIP_PATH, EXTRACT_PATH)

    # 2. Load data
    train_data, test_data = load_data(EXTRACT_PATH)

    # 3. Build model
    model = build_model(num_classes=train_data.num_classes)

    # 4. Train
    history = train_model(model, train_data, test_data)

    # 5. Evaluate & Plot
    evaluate_model(model, test_data)
    plot_metrics(history)

    # 6. Prediction
    predict_random(model, test_data)

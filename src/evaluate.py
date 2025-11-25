import matplotlib.pyplot as plt

def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

def plot_metrics(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

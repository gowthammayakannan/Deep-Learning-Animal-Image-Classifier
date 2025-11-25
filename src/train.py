def train_model(model, train_data, test_data, epochs=75):
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=test_data
    )
    return history

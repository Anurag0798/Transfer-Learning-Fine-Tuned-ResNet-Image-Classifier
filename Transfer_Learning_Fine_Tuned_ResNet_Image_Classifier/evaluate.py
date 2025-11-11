def evaluate_model(model, val_ds):
    loss, acc = model.evaluate(val_ds)
    print(f"Validation Accuracy: {acc*100:.2f}%")
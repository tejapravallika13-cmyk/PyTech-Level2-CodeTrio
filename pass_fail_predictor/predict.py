from train_model import train_model


def predict_result():
    model = train_model()

    if model is None:
        print("Model training failed.")
        return

    try:
        math = float(input("Enter Math score: "))
        reading = float(input("Enter Reading score: "))
        writing = float(input("Enter Writing score: "))
    except ValueError:
        print("Please enter valid numeric values only.")
        return

    prediction = model.predict([[math, reading, writing]])

    if prediction[0] == 1:
        print("Prediction: PASS ")
    else:
        print("Prediction: FAIL ")


if __name__ == "__main__":
    predict_result()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from preprocess import load_and_prepare_data, MAX_WORDS, MAX_LEN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

X_train_pad, X_test_pad, y_train, y_test, tokenizer = load_and_prepare_data()

model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        input_length=MAX_LEN
    ),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

loss, accuracy = model.evaluate(
    X_test_pad,
    y_test,
    verbose=0
)

print("Test Accuracy:", accuracy)

y_prob = model.predict(
    X_test_pad,
    verbose=0
)

y_pred = (y_prob > 0.5).astype(int).flatten()

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("sms_spam_lstm.keras")

import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")

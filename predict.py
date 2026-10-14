import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 50

model = load_model("sms_spam_lstm.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def predict_sms(message):

    cleaned = clean_text(message)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = float(
        model.predict(padded, verbose=0)[0][0]
    )

    if probability > 0.5:
        print("SMS:", message)
        print("Prediction: SPAM")
        print("Probability:", round(probability, 4))
    else:
        print("SMS:", message)
        print("Prediction: HAM")
        print("Probability:", round(1 - probability, 4))


predict_sms(
    "Congratulations! You have won a free prize. Claim now!"
)

predict_sms(
    "Hey, are you coming to college tomorrow?"
)

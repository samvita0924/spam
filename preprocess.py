import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_FILE = "spam_sms.csv"
MAX_WORDS = 5000
MAX_LEN = 50

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def load_and_prepare_data():

    df = pd.read_csv(DATA_FILE, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    else:
        raise ValueError("Dataset must contain v1/v2 or label/message columns.")

    df = df.dropna()

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna()
    df["label"] = df["label"].astype(int)

    if len(df) > 2000:
        df, _ = train_test_split(
            df,
            train_size=2000,
            stratify=df["label"],
            random_state=42
        )

    df["message"] = df["message"].apply(clean_text)

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer

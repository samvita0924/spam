# SMS Spam Detection Using LSTM

## Problem Statement
A mobile communication company wants to identify spam messages automatically using deep learning.

## Objective
Classify SMS messages as HAM (normal) or SPAM (unwanted).

## Dataset
SMS Spam Detection dataset with an approximately 2,000-message labelled subset.

## Tasks Completed
1. Load dataset and handle missing values.
2. Preprocess SMS text.
3. Tokenize and pad text sequences.
4. Build an LSTM model using TensorFlow/Keras.
5. Train and evaluate the model.
6. Calculate Accuracy, Precision, Recall and F1-Score.
7. Display the confusion matrix.
8. Predict unseen SMS messages.

## Model Architecture
Embedding → LSTM → Dropout → Dense with Sigmoid

## Why LSTM?
LSTM is suitable for sequential text data because it can learn useful dependencies between words in a message.

## Files
- SMS_Spam_Detection_LSTM.ipynb - Complete Colab implementation
- preprocess.py - Dataset loading and preprocessing
- train_model.py - LSTM training and evaluation
- predict.py - Unseen SMS prediction
- requirements.txt - Required Python libraries

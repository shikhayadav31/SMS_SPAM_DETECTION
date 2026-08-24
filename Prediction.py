import joblib
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
import re


def clean_text(text):
    text = re.sub(r'(https?://\S+|www\.\S+)',' URL ',text)
    text= str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ",text)
    return " ".join(text.split())
stemmer = PorterStemmer()
def stem_text(text):
    return " ".join(stemmer.stem(word) for word in text.split() )

word_tfidf = joblib.load("notebooks/models/word_tfidf.pkl")
char_tfidf = joblib.load("notebooks/models/char_tfidf.pkl")
nb_model = joblib.load("notebooks/models/naive_bayes.pkl")
svm_model = joblib.load("notebooks/models/svm.pkl")
lr_model = joblib.load("notebooks/models/logistic_regression.pkl")
rf_model = joblib.load("notebooks/models/random_forest.pkl")
char_weight=2.0

import tkinter as tk

def predict_message():
    message = message_box.get("1.0", tk.END).strip()

    if not message:
        result_label.config(text="Please enter a message.")
        return
    #Cleaning and Stemming
    cleaned_message = clean_text(message)
    cleaned_message = stem_text(cleaned_message)

    #TF-IDF Vectorization
    message_word = word_tfidf.transform([cleaned_message])
    message_char = char_tfidf.transform([cleaned_message]).multiply(char_weight)
    message_tfidf = hstack([
        message_word,
        message_char]).tocsr()

    #Predictions from all models
    predictions = [
        nb_model.predict(message_tfidf)[0],
        svm_model.predict(message_tfidf)[0],
        lr_model.predict(message_tfidf)[0],
        rf_model.predict(message_tfidf)[0]]
        

    spam_votes = sum(predictions)
    ham_votes = len(predictions) - spam_votes

    #Final prediction (in case of tie, svm is given priority)
    if spam_votes > ham_votes:
        final_result = "SPAM"
    elif ham_votes > spam_votes:
        final_result = "HAM"
    else:
        final_result = "SPAM" if predictions[1] == 1 else "HAM"

    result_label.config(
        text=f"Prediction: {final_result}" )

root = tk.Tk()
root.title("SMS Spam Detector")
root.geometry("600x450")

title_label = tk.Label(
    root,
    text="SMS Spam Detector",
    font=("Arial", 20, "bold"))
title_label.pack(pady=20)

label = tk.Label(
    root,
    text="Enter your message:",
    font=("Arial", 12))
label.pack()

message_box = tk.Text(
    root,
    height=6,
    width=60,
    font=("Arial", 12))
message_box.pack(pady=15)

check_button = tk.Button(
    root,
    text="Check Message",
    font=("Arial", 12, "bold"),
    padx=20,
    pady=8,
    command=predict_message)
check_button.pack(pady=10)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"))
result_label.pack(pady=15)

root.mainloop()


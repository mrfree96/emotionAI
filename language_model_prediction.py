import pandas as pd
import numpy as np
import joblib

pipe_lr = joblib.load(open("model/language_model.pkl", "rb"))

#emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
#                       "sadness": "😔", "surprise": "😮"}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def analyze_text(audio_name, text):
    raw_text = text
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)
    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
    proba_df_clean = proba_df.T.reset_index()
    proba_df_clean.columns = ["emotions", "probability"]
    proba_df.insert(loc=0, column='AudioName', value=audio_name)
    proba_df.insert(loc=1, column='Transcript', value=raw_text)
    return proba_df
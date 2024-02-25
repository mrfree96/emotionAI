import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import preprocessing
import pickle
import time
import matplotlib.pyplot as plt
import frame_extraction_from_video as extract_video
import model_prediction as mp
import openpyxl
import audio_extraction_from_video as aue
import speech_to_text as sttext
import language_model_prediction as lmp
import pandas as pd
import signal_model_prediction as signal_pred
from io import StringIO

def isnullcontrol(Emotion):
    # Get the value count of the given emotion, return 0 if the key does not exist
    emotion_count = data['Emotion'].value_counts().get(Emotion, 0)
    # Convert the count value to float
    emotion_count = float(emotion_count)
    # If the count value is null, return 0
    if pd.isnull(emotion_count):
        return 0
    # Return the count value
    return emotion_count


def calculate_end_result():
    d_angry = isnullcontrol('Angry')
    d_happiness = isnullcontrol('Happiness')
    d_neutral = isnullcontrol('Neutral')
    d_sad = isnullcontrol('Sad')
    d_fear = isnullcontrol('Fear')
    d_disgust = isnullcontrol('Disgust')
    d_surprise = isnullcontrol('Surprise')

    d_total = d_angry + d_sad + d_fear + d_surprise + d_disgust + d_neutral + d_happiness
    y_angry = (d_angry / d_total) * 50
    y_sad = (d_sad / d_total) * 50
    y_neutral = (d_neutral / d_total) * 50
    y_fear = (d_fear / d_total) * 50
    y_surprise = (d_surprise / d_total) * 50
    y_happiness = (d_happiness / d_total) * 50
    y_disgust = (d_disgust / d_total) * 50

    y_angry = y_angry + (float(language_prediction_data['anger'][0])) * 45
    y_sad = y_sad + (float(language_prediction_data['sadness'][0])) * 45
    y_neutral = y_neutral + (float(language_prediction_data['neutral'][0])) * 45
    y_fear = y_fear + (float(language_prediction_data['fear'][0])) * 45
    y_surprise = y_surprise + (float(language_prediction_data['surprise'][0])) * 45
    y_happiness = y_happiness + (float(language_prediction_data['happiness'][0])) * 45
    y_disgust = y_disgust + (float(language_prediction_data['disgust'][0])) * 45

    if (signal_data['Emotion'][0] == 'Angry'):
        y_angry = y_angry + 5
    elif (signal_data['Emotion'][0] == 'Sad'):
        y_sad = y_sad + 5
    elif (signal_data['Emotion'][0] == 'Neutral'):
        y_neutral = y_neutral + 5
    elif (signal_data['Emotion'][0] == 'Fear'):
        y_fear = y_fear + 5
    elif (signal_data['Emotion'][0] == 'Happiness'):
        y_happiness = y_happiness + 5
    elif (signal_data['Emotion'][0] == 'Surprise'):
        y_surprise = y_surprise + 5
    else:
        y_disgust = y_disgust + 5

    mydata = {'Angry': [y_angry], 'Sadness': [y_sad], 'Neutral': [y_neutral], 'Fear': [y_fear],
              'Disgust': [y_disgust], 'Happiness': [y_happiness], 'Surprise': [y_surprise]}
    results_df = pd.DataFrame(mydata)
    st.header("End Results", divider="rainbow")
    st.dataframe(results_df)
    for column in results_df.columns:
        if results_df[column].values[0] == results_df.max(axis=1).values:
            st.write('Final Decision: ' + column + '-->' + str(results_df.max(axis=1).values[0]))
    # Plotting the pie-chart
    # Create a figure and axis for the pie chart
    fig, ax = plt.subplots()
    wp = {'linewidth': 0.5, 'edgecolor': 'black'}
    # Create a pie chart with the given values
    ax.pie([results_df['Angry'].sum(), results_df['Sadness'].sum(), results_df['Neutral'].sum(),
            results_df['Fear'].sum(), results_df['Disgust'].sum(), results_df['Happiness'].sum(),
            results_df['Surprise'].sum()],
           labels=['Angry', 'Sadness', 'Neutral', 'Fear', 'Disgust', 'Happiness', 'Surprise'],
           explode=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
           autopct=lambda p: '{:.4f}%'.format(p),
           startangle=45,
           wedgeprops=wp,
           shadow=True)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    plt.title('Results')
    fig.legend(title='Emotions')
    st.pyplot(fig)


st.title(":rainbow[EmotionAI]")
st.header("", divider="rainbow")
st.write("\n:white[EmotionAI is an app designed for analyzing emotions from videos. This app utilizes 3 models."
         "The first model is based on facial expressions, the second model on speech, and the third model"
         " on audio signals. The final decision is determined by combining the results of these three models,"
         " each of which has different weight in the final outcome.]")
st.header("", divider="rainbow")

uploaded_file = st.file_uploader("Choose a video to analyse", type=['mp4'])

analyse_button = st.button("Analyse", type="primary")
# If file is uploaded
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile() as tempDir:
        tempDir.write(uploaded_file.getbuffer())
        if analyse_button:
            folder = extract_video.extract(tempDir.name)
            data = mp.predict_emotion(folder)
            st.header("Frame-by-Frame Emotion Detection: Analyzing Video Through Facial Expressions",
                      divider='rainbow')
            st.dataframe(data, use_container_width=True)

            audio_folder = aue.extract_audio(tempDir.name)
            transcription = sttext.convert_speech_to_text(audio_folder)
            st.header("Speech Recognition and Language Model Results",
                      divider="rainbow")
            st.header("Transcription", divider='blue')
            st.write(transcription)
            language_prediction_data = lmp.analyze_text(audio_folder, transcription)
            st.header("Language Model Results", divider='blue')
            language_prediction_data.drop(["AudioName", "Transcript"], axis=1, inplace=True)
            st.dataframe(language_prediction_data)

            signal_data = signal_pred.predict_signal_emotion(audio_folder)
            st.header("Audio Signal Model Results", divider='rainbow')
            st.dataframe(signal_data)

            #st.header("Conclusion:", divider="rainbow")
            calculate_end_result()
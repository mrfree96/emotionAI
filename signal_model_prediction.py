import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from os import path
from pydub import AudioSegment
import soundfile as sf
# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile
import os

def convert_to_spectogramv2(wav_file):
    # Read the WAV file
    Fs, aud = wavfile.read(wav_file)
    print(Fs, aud.shape)

    aud = aud[:, 0]  # Take only the first channel for mono
    print(aud.shape)
    # Generate the spectrogram
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs, NFFT=256)

    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    # Set the target directory for saving the spectrogram
    target_dir = r'/spectograms'

    # Extract the file name from the path
    spectrogram_name = wav_file.rsplit('/')[-1]

    # Construct the path for saving the spectrogram image
    spectrogram_path = target_dir+'/'+spectrogram_name+'.png'
    print(spectrogram_path)
    # Save the spectrogram image
    plt.savefig(str(spectrogram_path))
    # Return the path to the saved spectrogram
    return spectrogram_path


def convert_to_spectogram(vaw_file):
    Fs, aud = wavfile.read(vaw_file)
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
    plt.xticks([])
    plt.yticks([])
    target_dir = '/Users/a2023/PycharmProjects/grad_gui/spectograms'
    spectogram_name = vaw_file.rsplit('/')[-1]
    spect_path = target_dir+'/'+spectogram_name+'.png'
    plt.savefig(spect_path)
    return spect_path
def mp3_to_wav(mp3):
    # files
    dst = mp3+'.wav'
    # convert wav to mp3
    sound = AudioSegment.from_mp3(mp3)
    sound.export(dst, format="wav")
    return dst


def predict_signal_emotion(audio_path):

    new_model = tf.keras.models.load_model('model/signal_CNN.h5')

    results = []
    # bu kod bloğu metod içine yazılacaktır # to do
    label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Neutral', 5 : 'Sad', 6 : 'Surprise'}
    my_audio = audio_path
    # Iterate over all files in the folder
    my_wav = mp3_to_wav(my_audio)
    my_spectogram = convert_to_spectogramv2(my_wav)
    image = load_img(my_spectogram, target_size=(128, 96))
    image = img_to_array(image)
    print(image)
    img = np.expand_dims(image, 0)
    predictions = new_model.predict(img)
    predicted_class = np.argmax(predictions)
    results.append(label_dict[int(predicted_class)])
    df = pd.DataFrame({'Emotion': results})
    return df

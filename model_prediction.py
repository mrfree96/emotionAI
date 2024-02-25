import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array




def predict_emotion(folder_path):

    new_model = tf.keras.models.load_model('model/CNN.h5')

    frame_names = []
    timestamps = []
    results = []

    # bu kod bloğu metod içine yazılacaktır # to do
    label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Neutral', 5 : 'Sad', 6 : 'Surprise'}
    eval_dir = folder_path
    # Iterate over all files in the folder

    for filename in os.listdir(eval_dir):
        if filename.endswith('.jpg'):  # Add more extensions if needed
            # Create the full path to the image file
            img_path = os.path.join(eval_dir, filename)
            image = load_img(img_path, target_size=(48, 48))
            image = img_to_array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(gray, 0)
            predictions = new_model.predict(img)
            # Get the predicted class index
            predicted_class = np.argmax(predictions)
            plt.imshow(gray)
            # frame name adding to list
            frame_names.append(filename)
            split = filename.rsplit('_')
            timestamp = split[-2]
            # time stamp adding to list
            timestamps.append(timestamp)
            # emotion result adding to list
            results.append(label_dict[int(predicted_class)])
    df = pd.DataFrame({'FrameName': frame_names, 'TimeStamp': timestamps, 'Emotion': results})
    df_sorted = df.sort_values(by=['TimeStamp']).reset_index(drop=True)
    return df_sorted
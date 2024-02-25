import time
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext

import matplotlib.pyplot as plt

import frame_extraction_from_video as extract_video
import model_prediction as mp
import openpyxl
import audio_extraction_from_video as aue
import speech_to_text as sttext
import language_model_prediction as lmp
import pandas as pd
import signal_model_prediction as signal_pred
from tkinter import PhotoImage
class LogFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        # Create a Text widget for displaying log messages
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=40, height=25)
        self.log_text.pack(expand=False, fill=tk.BOTH)

    def log_message(self, message):
        # Set state back to 'normal' to make the written allowed
        self.log_text.config(state=tk.NORMAL)

        # Insert a log message into the Text widget
        self.log_text.insert(tk.END, message + '\n')

        # Set state back to 'disabled' to make the text read-only
        self.log_text.config(state=tk.DISABLED)

        # Automatically scroll to the bottom to show the latest message
        self.log_text.yview(tk.END)

text = ""

def browse_file(log_frame):
    global text
    file_path = filedialog.askopenfilename()
    text = file_path
    if text != "":
        log_frame.log_message(text + " video selected!")


def analyze_video(log_frame):
    if text != "":
        folder = extract_video.extract(text)
        log_frame.log_message('Extraction Process Finished to dir ='+folder)
        data = mp.predict_emotion(folder)
        # buraya rsplit ile videoname ile resuls eklenecek. # to do
        # data.to_excel('/Users/a2023/PycharmProjects/grad_gui/results/results.xlsx', index_label=False)
        audio_folder = aue.extract_audio(text)
        log_frame.log_message('Audio Extraction Finished: ' + audio_folder)
        transcription = sttext.convert_speech_to_text(audio_folder)
        log_frame.log_message('Transcription is:'+transcription)
        language_prediction_data = lmp.analyze_text(audio_folder, transcription)
        log_frame.log_message('Language-model Predicted your Transcription.')
        signal_data = signal_pred.predict_signal_emotion(audio_folder)
        log_frame.log_message('Audio-signal-model Predicted')
        # read
        # Calculation for results

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

        d_angry = isnullcontrol('Angry')
        d_happiness = isnullcontrol('Happiness')
        d_neutral = isnullcontrol('Neutral')
        d_sad = isnullcontrol('Sad')
        d_fear = isnullcontrol('Fear')
        d_disgust = isnullcontrol('Disgust')
        d_surprise = isnullcontrol('Surprise')

        d_total=d_angry+d_sad+d_fear+d_surprise+d_disgust+d_neutral+d_happiness
        y_angry=(d_angry/d_total)*50
        y_sad = (d_sad / d_total) * 50
        y_neutral = (d_neutral / d_total) * 50
        y_fear = (d_fear / d_total) * 50
        y_surprise = (d_surprise / d_total) * 50
        y_happiness = (d_happiness / d_total) * 50
        y_disgust = (d_disgust / d_total) * 50

        y_angry=y_angry+(float(language_prediction_data['anger'][0]))* 45
        y_sad = y_sad + (float(language_prediction_data['sadness'][0])) * 45
        y_neutral = y_neutral + (float(language_prediction_data['neutral'][0])) * 45
        y_fear = y_fear + (float(language_prediction_data['fear'][0])) * 45
        y_surprise = y_surprise + (float(language_prediction_data['surprise'][0])) * 45
        y_happiness = y_happiness + (float(language_prediction_data['happiness'][0])) * 45
        y_disgust = y_disgust + (float(language_prediction_data['disgust'][0])) * 45

        if(signal_data['Emotion'][0]=='Angry'):
            y_angry=y_angry+5
        elif(signal_data['Emotion'][0]=='Sad'):
            y_sad=y_sad+5
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

        mydata = {'Angry':[y_angry],'Sadness':[y_sad],'Neutral':[y_neutral],'Fear':[y_fear],
                  'Disgust':[y_disgust],'Happiness':[y_happiness],'Surprise':[y_surprise]}
        results_df=pd.DataFrame(mydata)
        log_frame.log_message("-"*80)
        log_frame.log_message("\nCALCULATED RESULTS")
        log_frame.log_message(str(results_df))
        for column in results_df.columns:
            if results_df[column].values[0] == results_df.max(axis=1).values:
                log_frame.log_message('Final Decision: '+column+'-->'+str(results_df.max(axis=1).values[0]))
        log_frame.log_message("-"*80)
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
        fig.savefig('results/results_pie.png')
        with pd.ExcelWriter('results/results.xlsx') as writer:
            data.to_excel(writer, sheet_name='FrameResults')
            language_prediction_data.to_excel(writer, sheet_name='TextResults')
            signal_data.to_excel(writer, sheet_name='AudioSignalResults')
            results_df.to_excel(writer, sheet_name='CalculatedResults')
            # one more sheet for display end results

    else:
        msg = "Please select a file!"
        messagebox.showerror("Error", msg)

# Create the main window
root = tk.Tk()

log_frame = LogFrame(master=root)
log_frame.pack(expand=True, fill=tk.BOTH)

root.geometry('800x500')
root.title("Psychological Analysis")




# Create a button to browse for a file
browse_button = tk.Button(root, text="Browse Video", command=lambda: browse_file(log_frame))
browse_button.pack(pady=10)


analyze_button = tk.Button(root, text="Analyse Video", command=lambda: analyze_video(log_frame))
analyze_button.pack(pady=10)



# Run the Tkinter event loop
root.mainloop()
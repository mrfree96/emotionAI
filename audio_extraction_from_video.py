import moviepy.editor as me
import os

def create_folder(folder_name):
    # Specify the path of the folder
    folder_path = folder_name

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    return folder_path


def extract_audio(video_file):

    video = me.VideoFileClip(video_file)

    audio = video.audio

    split = video_file.rsplit('/')[-1]

    audio_path = '/audios/' + split
    folder_path = create_folder(audio_path)
    video_name = audio_path.rsplit('.')[-2]
    video_name = video_name.rsplit('/')[-1]
    audio_file = folder_path+'/'+video_name + '.mp3'

    audio.write_audiofile(audio_file)

    return audio_file

import moviepy.editor as me
import os
import tempfile

"""
def create_folder(folder_name):
    # Specify the path of the folder
    folder_path = folder_name

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    return folder_path
"""

def create_folder(base_path, folder_name):
    # Use a writable base path, like a temp directory or user's home directory
    writable_base_path = os.path.join(base_path, folder_name)
    os.makedirs(writable_base_path, exist_ok=True)
    return writable_base_path

def extract_file_name(audio_path):
    # Use os.path.splitext which safely splits the extension and returns the file name without the extension
    file_name_without_extension, _ = os.path.splitext(os.path.basename(audio_path))
    return file_name_without_extension

def extract_audio(video_file):

    base_path = tempfile.gettempdir()
    folder_name = 'audios'
    folder_path = create_folder(base_path, folder_name)

    video = me.VideoFileClip(video_file)

    audio = video.audio

    split = video_file.rsplit('/')[-1]

    audio_path = split
    folder_path = create_folder(folder_path, audio_path)

    #video_name = audio_path.rsplit('.')[-2]
    #video_name = video_name.rsplit('/')[-1]
    video_name = extract_file_name(folder_path)

    audio_file = folder_path+'/'+video_name + '.mp3'

    audio.write_audiofile(audio_file)

    return audio_file

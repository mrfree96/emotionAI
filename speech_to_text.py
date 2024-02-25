import whisper

# This method returns transcript text of given audio
def convert_speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]
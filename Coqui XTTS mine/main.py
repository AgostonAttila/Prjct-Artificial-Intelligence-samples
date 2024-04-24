from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="this is an english text, i am an AI assistant",
                file_path="output.wav",
                speaker_wav=["ref1.wav"],
                language="en",
                split_sentences=True
                )
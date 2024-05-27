import whisper
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import soundfile as sf

# Whisper modell betöltése
model = whisper.load_model("base")

# Hangfájl transzkripciója
result = model.transcribe("magyarpamkutya.wav")
transcription = result["text"]

# A Wav2Vec2 modell betöltése hangszintézishez
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Szöveg konvertálása tokenekké
input_values = tokenizer(transcription, return_tensors="pt").input_values

# A tokenek konvertálása hanggá
with torch.no_grad():
    logits = model(input_values).logits

# A legvalószínűbb tokent kiválasztása
predicted_ids = torch.argmax(logits, dim=-1)

# A tokenek visszakonvertálása szöveggé
transcription = tokenizer.batch_decode(predicted_ids)[0]

# Az eredmény mentése hangfájlba
sf.write("output_audio_file.wav", transcription, 16000)

print("Hangklónozás kész!")
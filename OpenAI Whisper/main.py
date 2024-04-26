import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import torch
from IPython.display import Audio as display_Audio, display
import torchaudio

#English transcription full example from https://huggingface.co/openai/whisper-large-v3
device = torch.device('cpu')
torch_dtype = torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
result = pipe(sample)
print(result["text"])


#utility functions
def load_recorded_audio(path_audio,input_sample_rate=48000,output_sample_rate=16000):
    # Dataset: convert recorded audio to vector
    waveform, sample_rate = torchaudio.load(path_audio)
    waveform_resampled = torchaudio.functional.resample(waveform, orig_freq=input_sample_rate, new_freq=output_sample_rate) #change sample rate to 16000 to match training. 
    sample = waveform_resampled.numpy()[0]
    return sample

def run_inference(path_audio, output_lang, pipe):
    sample = load_recorded_audio(path_audio)
    result = pipe(sample, generate_kwargs = {"language": output_lang, "task": "transcribe"})
    print(result["text"])

path_audio = "sample.wav"
output_lang = "en"
run_inference(path_audio,output_lang, pipe)

path_audio = "magyar.wav"
output_lang = "hu"
run_inference(path_audio,output_lang, pipe)


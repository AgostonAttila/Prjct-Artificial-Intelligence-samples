from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from IPython.display import Audio
from scipy.io.wavfile import write

config = XttsConfig()
config.load_json("./XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/")
model.cuda()

text_to_speak = "We are going to go to the gym now to lift heavy circles."
reference_audios = ["./sample_voice1.wav","./sample_voice2.wav"]

outputs = model.synthesize(
    text_to_speak,
    config,
    speaker_wav=reference_audios,
    gpt_cond_len=3,
    language="en",
)

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference_audios)

outputs = model.inference(
    text=string,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    language="en",
    enable_text_splitting=True
)

Audio(data=outputs['wav'], rate=24000)

output_file_path = f'./outputs/output_audio.wav'
write(output_file_path, 24000, outputs['wav'])
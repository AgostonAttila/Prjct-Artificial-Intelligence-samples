from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="Ez egy beszéd teszt, egy kettő három Bár még elég sok vizsgálatra van szükség, hogy a tudósok megállapítsák, a 40 fényévnyire felfedezett Gliese–12b exobolygó vajon lakható-e, az előzetes eredmények eléggé meggyőzőek. Legalábbis a hőmérséklet nagyjából stimmel, és hogyha olyan légkörrel rendelkezik, ami a víz kialakulását – és így az életet – lehetővé teszi, akkor könnyen lehet, hogy megvan, hová költözhet az emberiség a Föld után.",
                file_path="output.wav",
                speaker_wav=["magyarpamkutya.wav"],
                language="hu",
                split_sentences=True
                )



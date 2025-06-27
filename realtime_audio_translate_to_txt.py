# Traduction audio en temps réel depuis la sortie audio/micro et écriture dans un fichier texte
# Nécessite : sounddevice, numpy, openai-whisper, googletrans==4.0.0-rc1, torch
# python 3.12.0

import sounddevice as sd
import soundfile as sf
import whisper
from googletrans import Translator
import os
import time
import warnings
import asyncio
from typing import Any

class RealTimeAudioTranscriber:
    def __init__(self, 
                 device: int = 12, 
                 channels: int = 2, 
                 samplerate: int = 48000, 
                 blocksize: int = 2048, 
                 duration: int = 5, 
                 language: str = "fr", 
                 destination: str = "en",
                 model_size: str = "base"):
        self.device: int = device
        self.channels: int = channels
        self.samplerate: int = samplerate
        self.blocksize: int = blocksize
        self.duration: int = duration
        self.language: str = language
        self.destination: str = destination
        self.model: Any = whisper.load_model(model_size)
        self.translator: Translator = Translator()
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

    async def run(self) -> None:
        with sd.InputStream(
            device=self.device,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32'
        ) as stream:
            print("Enregistrement du son des haut-parleurs... Ctrl+C pour arrêter.")
            try:
                while True:
                    audio_chunk, _ = stream.read(int(self.duration * self.samplerate))
                    print("Chunk capturé, shape:", audio_chunk.shape)
                    filename = self.save_chunk(audio_chunk)
                    text = self.transcribe(filename)
                    print("Transcription :", text)
                    translated = await self.translate_text(text)
                    print("Traduction :", translated)
                    self.write_in_file(translated)
                    os.system(f"del {filename}")
            except KeyboardInterrupt:
                print("Arrêt de l'enregistrement.")

    def save_chunk(self, chunk) -> str:
        filename: str = f'temp_realtime_{int(time.time())}.wav'
        sf.write(filename, chunk, self.samplerate, subtype='FLOAT')
        print(f"Chunk sauvegardé dans {filename}")
        return filename

    def transcribe(self, filename: str) -> str:
        # Mettre fp16 à true si vous avez une carte graphique Nvidia qui supporte cuda
        result = self.model.transcribe(filename, language=self.language, fp16=False)
        return result["text"]

    async def translate_text(self, text: str) -> str:
        result = await self.translator.translate(text, dest=self.destination)
        return result.text

    def write_in_file(self, text: str) -> None:
        filename: str = "test.txt"
        with open(filename, "w+", encoding="utf-8") as f:
            f.write(text)
        print(f"Traduction sauvegardée dans {filename}")

    async def test(self) -> None:
        # Nom de l'audio pour les testes du modèle
        print(sd.query_devices())
        filename: str = "test.wav"
        data, samplerate = sf.read(filename)
        sd.play(data, samplerate)
        sd.wait()
        transcribe: str = self.transcribe(filename)
        translate: str = await self.translate_text(transcribe)
        self.write_in_file(translate)


if __name__ == '__main__':
    transcriber = RealTimeAudioTranscriber(
        device=14,          # À adapter selon ton périphérique cherche wasapi sd.query_devices()
        channels=2,         # 1 Pour Mono et 2 pour stéréo le mieux reste stéréo
        samplerate=48000,   # A voir en fonction de ce que supporte wasapi sd.querydevice(device numéro)
        blocksize=2048,     # taille de chaque échantillon
        duration=5,         # La durée de chaque vidéo généré
        language="fr",      # langage de la vidéo
        destination="en",   # langage à traduire
        model_size="base"   # taille du modèle tiny, base, small, medium, large
    )
    asyncio.run(transcriber.run())
    # Pour les testes
    #asyncio.run(transcriber.test())
from aiogram import Bot, Dispatcher, executor, types

import scipy
import numpy as np

import librosa
import soundfile as sf
from pydub import AudioSegment

import os
from time import time, sleep

from model import *

import warnings


API_TOKEN = '6336207323:AAFtlMAE3v5JycQORaBm1-lE1RIMLV-uyE8'


bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(content_types=['voice'])
async def voice_processing(message):
    f = str(time())

    file = await bot.get_file(message.voice.file_id)
    await bot.download_file(file.file_path, f"{f}.mp3")

    y, sr = librosa.load(f'{f}.mp3', mono=True)

    features = []
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  # mfcc_mean<0..20>
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  # mfcc_std
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])  # cent_mean
    features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])  # cent_std
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0])  # cent_skew
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[0])  # rolloff_mean
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[0])  # rolloff_std

    os.remove(f'{f}.mp3')

    q = model.predict(stab.transform([features])[0])
    if q == 'angry':
        await bot.send_message(message['from'].id, f"Похоже, вы злитесь...")
        await bot.send_message(message['from'].id, "😡")
    elif q == 'sad':
        await bot.send_message(message['from'].id, "Вы, видимо, расстроены, судя по голосу...")
        await bot.send_message(message['from'].id, "😭")


@dp.message_handler(commands=['help'])
async def send_welcome(message: types.Message):
    await bot.send_message(message['from'].id, "Я бот для определения эмоций по голосу! Просто отправьте мне аудиосообщение.")


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await bot.send_message(message['from'].id, "Привет! Я бот для определения эмоций по голосу! Просто отправьте мне аудиосообщение.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

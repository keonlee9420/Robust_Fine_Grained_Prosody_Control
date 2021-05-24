import numpy as np
from scipy.io.wavfile import read
import librosa
from scipy.io import wavfile
from scipy import interpolate
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool() # (B, max_len)
    return mask


def load_wav_to_torch(full_path):
    NEW_SAMPLERATE = 22050
    old_samplerate, old_audio = read(full_path)

    if old_samplerate != NEW_SAMPLERATE:
        duration = old_audio.shape[0] / old_samplerate

        time_old = np.linspace(0, duration, old_audio.shape[0])
        time_new = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

        interpolator = interpolate.interp1d(time_old, old_audio.T)
        new_audio = interpolator(time_new).T

        wavfile.write(full_path, NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))

    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    data, sampling_rate = librosa.load(full_path, sr=22050)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_torch_new(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

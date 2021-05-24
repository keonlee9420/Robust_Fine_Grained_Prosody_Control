# ignore tensorflow depreciate warnings
# see: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import sys
sys.path.append('waveglow/')
import numpy as np
import pandas as pd
import time
import torch
import argparse

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
from waveglow.denoiser import Denoiser
from waveglow.mel2samp import files_to_list, MAX_WAV_VALUE

from itertools import cycle
from data_utils import TextMelLoader, TextMelCollate
from utils import load_wav_to_torch
import textwrap

def make_space_above(axes, topmargin=1): # see https://stackoverflow.com/a/55768955
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def plot_data(data, transcript, image_path, figsize=(20, 4)):
    print("plot results...")
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    fig_names = ['reference', 'output', 'ref_alignment', 'alignment']
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
        axes[i].set_xlabel(fig_names[i])
    plt.suptitle("\n".join(textwrap.wrap(transcript, 130))) # see https://stackoverflow.com/a/55768955
    make_space_above(axes, topmargin=1)
    plt.savefig(image_path)
    print("All plots saved!: %s" % image_path)

    plt.close()

def load_mel(hparams, stft, reference_audio_path):
    audio, sampling_rate = load_wav_to_torch(reference_audio_path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec

def prepare_speaker_set(hparams):
    # Define Speakers Set
    speaker_ids = TextMelLoader("filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist_skipped.txt", hparams).speaker_ids
    speakers = pd.read_csv('filelists/libritts_speakerinfo.txt', engine='python',header=None, comment=';', sep=' *\| *',
                        names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
    speakers['MELLOTRON_ID'] = speakers['ID'].apply(lambda x: speaker_ids[x] if x in speaker_ids else -1)
    female_speakers = cycle(
        speakers.query("SEX == 'F' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
    male_speakers = cycle(
        speakers.query("SEX == 'M' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
    return female_speakers, male_speakers

def get_sample_speaker_id(hparams):
    female_speakers, male_speakers = prepare_speaker_set(hparams)
    speaker_id = next(male_speakers) # next(female_speakers) if np.random.randint(2) else next(male_speakers)
    speaker_id = torch.LongTensor([speaker_id]).cuda()
    return speaker_id

def synthesize(hparams, model, waveglow, stft, outdir, transcript, reference_audio_path, speaker_id, filename=None):
    filename = filename if filename else reference_audio_path.split('/')[-1].replace('.wav', '') + "_Robust_Fine_Grained_tacotron2"

    sequence = np.array(text_to_sequence(transcript, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    reference_mel = load_mel(hparams, stft, reference_audio_path)

    with torch.no_grad():
        output_mel_path = os.path.join(
            outdir, "{}.png".format(filename))
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference((sequence, reference_mel, speaker_id)) # should be running under no_grad
        alignments, ref_alignments = alignments
        plot_data((reference_mel.float().data.cpu().numpy()[0],
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                ref_alignments.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T),
                transcript,
                output_mel_path)

        print("infer audio...")
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        # audio = denoiser(audio, .0)
        audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        output_audio_path = os.path.join(
            outdir, "{}.wav".format(filename))

        write(output_audio_path, hparams.sampling_rate, audio)
        # print("Synthesized audio saved!: %s" % output_audio_path)
    print("\n")

def load_models(hparams, checkpoint_path, waveglow_path):
    print("load models...")
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.cuda().eval()

    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    print("loaded!")

    return model, waveglow

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                    required=True, help='checkpoint path')
    parser.add_argument('-t', '--text', type=str,
                    required=True, help='text to synthesize')
    parser.add_argument('-r', '--reference_audio_path', type=str,
                    required=True, help='audio path for reference (style transfer)')
    parser.add_argument('-w', '--waveglow_path', type=str,
                    required=False, help='waveglow path',
                    default='/home/mcm/pyprojects/Robust_Fine_Grained_Prosody_Control/waveglow/waveglow_256channels.pt')
    args = parser.parse_args()
    transcript = args.text
    checkpoint_path = args.checkpoint_path
    reference_audio_path = args.reference_audio_path
    waveglow_path = args.waveglow_path

    assert os.path.isfile(checkpoint_path), "No such checkpoint"
    assert os.path.isfile(reference_audio_path), "No such reference audio"
    assert os.path.isfile(waveglow_path), "No such waveglow"

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    outdir = os.path.join("results", "single_inference")
    hash_ = '{0:010x}'.format(int(time.time() * 256))[:10]
    filename = hash_ + '_' + reference_audio_path.split('/')[-1].replace('.wav', '') + '_BY_' + checkpoint_path.split('/')[-1]

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model, waveglow = load_models(hparams, checkpoint_path, waveglow_path)

    stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    speaker_id = get_sample_speaker_id(hparams)

    print("------------- inference -------------")
    print("input text: \n%s" % transcript)
    synthesize(hparams, model, waveglow, stft, outdir, transcript, reference_audio_path, speaker_id, filename)
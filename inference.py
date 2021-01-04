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

def plot_data(data, transcript, image_path, figsize=(11, 4)):
    print("plot results...")
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    fig_names = ['output', 'alignment']
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
        axes[i].set_xlabel(fig_names[i])
    plt.suptitle("\n".join(textwrap.wrap(transcript, 100))) # see https://stackoverflow.com/a/55768955
    make_space_above(axes, topmargin=1)
    plt.savefig(image_path)
    print("All plots saved!: %s" % image_path)
    
    plt.close()

def synthesize(hparams, model, waveglow, outdir, transcript, filename):
    sequence = np.array(text_to_sequence(transcript, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    with torch.no_grad():
        output_mel_path = os.path.join(
            outdir, "{}.png".format(filename))
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
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
        audio_path = os.path.join(
            outdir, "{}.wav".format(filename))
        
        write(audio_path, hparams.sampling_rate, audio)
        # print("Synthesized audio saved!: %s" % audio_path)
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
    parser.add_argument('-w', '--waveglow_path', type=str,
                    required=False, help='waveglow path',
                    default='/home/keon/contextron/pretrained_models/waveglow_256channels_universal_v5.pt')
    args = parser.parse_args()
    transcript = args.text
    checkpoint_path = args.checkpoint_path
    waveglow_path = args.waveglow_path

    assert os.path.isfile(checkpoint_path), "No such checkpoint"
    assert os.path.isfile(waveglow_path), "No such waveglow"

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    outdir = os.path.join("results", "single_inference")
    hash_ = '{0:010x}'.format(int(time.time() * 256))[:10]
    filename = hash_ + '_' + checkpoint_path.split('/')[-1]

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    model, waveglow = load_models(hparams, checkpoint_path, waveglow_path)

    print("------------- inference -------------")
    print("input text: \n%s" % transcript)
    synthesize(hparams, model, waveglow, outdir, transcript, filename)
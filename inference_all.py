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

from inference import synthesize, load_models

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                    required=True, help='checkpoint path')
    parser.add_argument('-f', '--filelist_path', type=str, default='datasets/refs_bc_parallel/refs_bc_parallel.txt',
                        required=False, help='path to filelist.txt)')
    parser.add_argument('-w', '--waveglow_path', type=str,
                    required=False, help='waveglow path',
                    default='/home/keon/contextron/pretrained_models/waveglow_256channels_universal_v5.pt')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    waveglow_path = args.waveglow_path
    filelist_path = args.filelist_path

    assert os.path.isfile(checkpoint_path), "No such checkpoint"
    assert os.path.isfile(waveglow_path), "No such waveglow"
    assert os.path.isfile(filelist_path), "No such filelist"

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    outdir = os.path.join("results", checkpoint_path.split('/')[-1], filelist_path.split('/')[-1].replace('.txt','_txt'))

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model, waveglow = load_models(hparams, checkpoint_path, waveglow_path)

    with open(filelist_path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            transcript, filename = line.split('|')[1], line.split('|')[0].split('/')[-1].replace('.wav', '') + "_bc13_tacotron2"
            print("\n------------- {} inference -------------".format(i+1))
            print("input text: \n%s" % transcript.replace('\n', ''))
            synthesize(hparams, model, waveglow, outdir, transcript, filename)
    print("outdir:", outdir)
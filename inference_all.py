# ignore tensorflow depreciate warnings
# see: https://github.com/tensorflow/tensorflow/issues/27045#issuecomment-480691244
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

import os
import torch
import argparse

from hparams import create_hparams
from layers import TacotronSTFT

from inference import synthesize, load_models, get_sample_speaker_id

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                    required=True, help='checkpoint path')
    parser.add_argument('-r', '--references_path', type=str, default='datasets/refs_bc_parallel',
                        required=False, help='path to references directory(should contains both wave files and filelist.txt)')
    parser.add_argument('-w', '--waveglow_path', type=str,
                    required=False, help='waveglow path',
                    default='/home/keon/contextron/pretrained_models/waveglow_256channels_universal_v5.pt')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    waveglow_path = args.waveglow_path
    references_path = args.references_path
    references_filelist_path = os.path.join(references_path, '{}.txt'.format(references_path.split('/')[-1]))

    assert os.path.isfile(checkpoint_path), "No such checkpoint"
    assert os.path.isfile(waveglow_path), "No such waveglow"
    assert os.path.isdir(references_path), "No such references path"

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    outdir = os.path.join("results", checkpoint_path.split('/')[-1], references_path.split('/')[-1])

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    model, waveglow = load_models(hparams, checkpoint_path, waveglow_path)

    stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    speaker_id = get_sample_speaker_id(hparams)

    with open(references_filelist_path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            transcript, reference_audio_path = line.split('|')[1], line.split('|')[0]
            print("\n------------- {} inference -------------".format(i+1))
            print("input text: \n%s" % transcript.replace('\n', ''))
            synthesize(hparams, model, waveglow, stft, outdir, transcript, reference_audio_path, speaker_id)
    print("outdir:", outdir)
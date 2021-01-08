# Robust and fine-grained prosody control of end-to-end speech synthesis (with waveglow)

Pytorch Implementation of [Robust and fine-grained prosody control of end-to-end speech synthesis](https://arxiv.org/abs/1811.02122) (Unofficial)

This implementation uses the [LibriTTS dataset](https://openslr.org/60/).

## Notes
1. *dev* branch: Tacotron2 with multispeaker (speaker embedding). Speaker information is only consumed by Decoder module, and Attention module doesn't see any of it (as authors' intention).
2. *text_side* branch: Text-side prosody control model implementation.
3. Speech-side prosody control and Prosody normalization are not implemented in current version, but you can simply add them on top of above branches.

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LibriTTS dataset](https://openslr.org/60/)
2. Clone this repo: `git clone https://github.com/keonlee9420/Robust_Fine_Grained_Prosody_Control.git`
3. CD into this repo: `cd Robust_Fine_Grained_Prosody_Control`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,/home/keon/speech-datasets/LibriTTS_preprocessed/train-clean-100/,your_libritts_dataset_folder/,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths 
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
(TBD)

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. Not supported in current implementation.

## Inference demo
1. Single sample: `python inference.py -c checkpoint/path -r reference_audio/wav/path -t "synthesize text"`
4. Multi samples: `python inference_all.py -c checkpoint/path -r reference_audios/dir/path`

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [NVIDIA/Tacotron-2](ttps://github.com/NVIDIA/tacotron2), [KinglittleQ/GST-Tacotron](https://github.com/KinglittleQ/GST-Tacotron)

We are thankful to the paper authors, specially Younggun Lee, and Taesu Kim.

[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
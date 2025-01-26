# [atet](https://github.com/atet) / [**_tts_**](https://github.com/atet/tts/blob/main/README.md#atet--tts)

[![.img/logo_tts.jpg](.img/logo_tts.jpg)](#nolink)

Technically, we are starting with an audio clip and continuing from that to generate new audio, speech-text-to-speech, speech-to-speech? *Whatever...*

----------------------------------------------------------------------------

## Table of Contents

* [0. Requirements](#0-requirements)
* [1. Docker](#1-docker)
* [2. Installation](#2-installation)
* [3. Basic Examples](#3-basic-examples)
* [4. Next Steps](#4-next-steps)

### Supplemental

* [Other Resources](#other-resources)
* [Troubleshooting](#troubleshooting)
* [References](#references)

----------------------------------------------------------------------------

## 0. Requirements

A bunch of requirements, but at least they're all free (except for that GPU if you don't already have one):

- Free Hugging Face account at: https://huggingface.co/join
- High-speed internet, as you'll need to download almost 25 GB of data
- Short ~10-15 second audio clip with transcript (speech-to-text not covered here)
- Windows Subsystem for Linux (WSL) with Docker (and NVIDIA Container Toolkit if using NVIDIA GPU), more info here:
   - [Installing WSL and Docker](https://github.com/atet/wsl)
   - [Installing NVIDIA Container Toolkit](https://github.com/atet/llm?tab=readme-ov-file#2-installation)
- (Optional) An NVIDIA GPU with at least 20 GB of VRAM as CPU processing is ***much, much slower***:

Mode | Execution Time (Mins.)
--- | ---
CPU | 🐌 10
GPU | 🚀 1

### Models From Hugging Face

- You must agree to `HKUSTAudio/Llasa-3B` repository terms on Hugging Face website before you can clone it<sup>1</sup>
- Three repositories being downloaded (~24 GB total):
   - [`HKUSTAudio/Llasa-3B` (~8 GB)](https://huggingface.co/HKUSTAudio/Llasa-3B)
   - [`HKUSTAudio/xcodec2` (~11 GB)](https://huggingface.co/HKUSTAudio/xcodec2)
   - [`facebook/w2v-bert-2.0` (~5 GB)](https://huggingface.co/facebook/w2v-bert-2.0)
- Download them to your WSL home directory, this may take a while ☕:

```bash
$ mkdir -p ~/models/HKUSTAudio && cd ~/models/HKUSTAudio && \
  git lfs clone git@hf.co:HKUSTAudio/Llasa-3B && \
  git lfs clone git@hf.co:HKUSTAudio/xcodec2 && \
  mkdir -p ~/models/facebook && cd ~/models/facebook && \
  git lfs clone git@hf.co:facebook/w2v-bert-2.0
```

### Input Audio

- Audio must be at 16 kHz sample rate, mono (not stereo), `*.wav` format
   - Free audio conversion with Audacity program: https://portableapps.com/apps/music_video/audacity_portable
   - Audio used for input should be around 15 seconds and expected output to be about 15 seconds of speech
   - Entire prompt + newly-generated audio can only be about 35 seconds long with this model:
      - Longer prompt audio (15-20 seconds) allows for better voice mimicking but shorter generated audio (15-10 seconds)
      - Shorter prompt audio (~10 seconds) allows longer generated audio (~25 seconds) but worse voice mimicking
- Example input audio files:
   - Public domain (CC0) voice clip from: https://opengameart.org/content/airport-announcement-voice-acting-stk
      - This file is located in this repository: `./.dat/voice.wav`
   - Machine-generated voice example from [www.morgbob.com](https://www.morgbob.com) ([Microsoft Text-to-Speech](https://learn.microsoft.com/en-us/answers/questions/1192398/can-i-use-azure-text-to-speech-for-commercial-usag#:~:text=%40Newstart%20Yes%2C%20you%20can%20use,mentioned%20in%20the%20pricing%20page.))
      - This file is located in this repository: `./.dat/morgbob.wav`

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## 1. Docker

We will make a custom Docker image that has CUDA, Pytorch, Jupyter Labs, and required Python dependencies to run the text to speech code.

### Creating Custom Image

- Pull `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image and create a dockerfile:

```bash
$ docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel && \
  mkdir -p ~/docker && cd ~/docker && \
  nano dockerfile
```

- Copy and paste this dockerfile:

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /root

RUN apt update
RUN apt -y upgrade
RUN apt -y install nano tmux htop rsync curl git

RUN pip install jupyterlab ipywidgets xcodec2==0.1.3
EXPOSE 8888
```

- Build the image, this may take a while ☕:

```bash
$ docker build -t tts_image .
```

- Create a Docker container and log into it:
   - `--gpus all` is a required flag for NVIDIA GPU processing
   - Must have absolute path to your home directory in WSL

```bash
$ docker run -dit --gpus all -p 8888:8888 -v <ABSOLUTE_PATH_TO_HOME>:/root/shared --name tts tts_image && \
  docker exec -it tts /bin/bash
```

- Copy models into Docker container (for much faster loading within container):

```bash
# mkdir -p /root/models/HKUSTAudio && cd /root/models/HKUSTAudio && \
  rsync --progress -r /root/shared/models/HKUSTAudio/Llasa-3B /root/models/HKUSTAudio/ && \
  rsync --progress -r /root/shared/models/HKUSTAudio/xcodec2 /root/models/HKUSTAudio/ && \
  mkdir -p /root/models/facebook && cd /root/models/facebook && \
  rsync --progress -r /root/shared/models/facebook/w2v-bert-2.0 /root/models/facebook/
```

- **IMPORTANT**: You must change the path for `facebook/w2v-bert-2.0` to point to your local repository for this model:

```bash
# cp /opt/conda/lib/python3.11/site-packages/xcodec2/modeling_xcodec2.py /opt/conda/lib/python3.11/site-packages/xcodec2/modeling_xcodec2.py.BAK && \
  sed -i 's/facebook\/w2v-bert-2.0/\/root\/shared\/models\/facebook\/w2v-bert-2.0/g' /opt/conda/lib/python3.11/site-packages/xcodec2/modeling_xcodec2.py
```

- Start Jupyter Lab server:

```bash
# cd /root && \
  jupyter lab --port-retries=0 --ip 0.0.0.0 --allow-root --ServerApp.token=""
```

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## 2. Installation

INSTALLATION.

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## 3. Basic Examples

BASIC EXAMPLES.

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## 4. Next Steps

NEXT STEPS.

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## Other Resources

**Description** | **URL Link**
--- | ---
null | null

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## Troubleshooting

Issue | Solution
--- | ---
**"It's not working!"** | This concise tutorial has distilled hours of sweat, tears, and troubleshooting; _it can't not work_

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

## References

1. Llasa 3B is licensed under [Creative Commons Attribution Non Commercial No Derivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en):
   - **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
   - **Non Commercial** — You may not use the material for commercial purposes.
   - **No Derivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

[Back to Top](#table-of-contents)

----------------------------------------------------------------------------

<p align="center">Copyright © 2025-∞ Athit Kao, <a href="http://www.athitkao.com/tos.html" target="_blank">Terms and Conditions</a></p>
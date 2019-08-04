# AudioFraud

This is a Senior Design Audio Fraud Project that will focus on differentiations provided from extraction of audio feautures to determine whether the speaker is authentic or computer generated all while identifying the users in the process.

## Problem:

Currently, there is not a system in place to detect fraudulent computer generated audio. Yet, there are systems that is able to take a 3-5 second snippet of your voice and could generate full length audio using your voice. This causes security issues since human biometrics are being used to access secure information.

## Solution:

Design a secure voice recognition system that is able to detect computer generated and recorded audio fraud.

## Installation:

```bash
git clone git@github.com:ylester/AudioFraud.
cd AudioFraud
python3 -m virtualenv --python=$pypath .venv
source .venv/bin/activate
pip3 install -r requirements.txt 
```

## Modules:

- Fraud Detection Algorithm
    - This module will focus on detecting fraud (computer generated audio) within a time span of 3-5 seconds.

- Speaker Recognition Algorithm
    - The algorithm will identify the slight alterations given by the different set of audios to detect the current speaker.

- Wifi Module Connection
    - Will handle data flow to and from the wifi module

![Screen_Output](https://raw.githubusercontent.com/ylester/AudioFraud/master/SMD.png)


### Team Members:

- [Yesha Lester](https://github.com/ylester)

- [Sedrick Cashaw Jr](https://github.com/sedcash)

- [Hussein (Shawn) El-Souri](https://github.com/helsouri)

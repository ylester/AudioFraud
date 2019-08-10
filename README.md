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


![Screen_Output](https://raw.githubusercontent.com/ylester/AudioFraud/master/SMD.png)

- Hardware
	- The hardware is now very close to being an "all in one" via the usage of rasberry pi that can directly program and run arduinio
	- Discovered that it will be hastle to fun i2c devices directly from arduino instead configured pi to send UART commands/data to redboard and will program i2c devices from there
	- python script of host and server now able to send files
	- python audio recording script perfect with using comman line commands

### Team Members:

- [Yesha Lester](https://github.com/ylester)

- [Sedrick Cashaw Jr](https://github.com/sedcash)

- [Hussein (Shawn) El-Souri](https://github.com/helsouri)


### References :
- https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone

- for recording audio the python script is unreliable overflowing for reaosn i do not know so there is a terminal command to do it without having to do any python arecord -D hw:1,0 -d 10 -f cd test.wav -c 1
added functions to run lcd and LED on python and socket script now works



import speech_recognition as sr

# Instance of the Recognizer class
r = sr.Recognizer()

# Captures data from an audio recording
# In case we decide tp save to a file and then analyze
voice = sr.AudioFile('voice.wav')
with voice as source:
    audio = r.record(source)

r.recognize_google(audio_data=audio)

# Code to read from a Microphone
mic = sr.Microphone()

""" If your system has no default microphone (such as on a RaspberryPi), 
or you want to use a microphone other than the default, you will need to 
specify which one to use by supplying a device index. You can get a list 
of microphone names by calling the list_microphone_names() static method 
of the Microphone class."""

# Displays the list of available microphones
sr.Microphone.list_microphone_names()
# mic = sr.Microphone(device_index=3)

with mic as source:
    audio = r.listen(source=source)

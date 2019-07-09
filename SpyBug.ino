#include <pcmConfig.h>
#include <pcmRF.h>
#include <TMRpcm.h>
#include <SPI.h>
#include <SD.h>

#define PIN_GATE_IN 2
// setup to try to blink on borad LED
// Serial.begin(9600); to initiate COM
// Serial.println("Initialized");
/*
Steps:
1. Edit pcmConfig.h
    a: On Uno or non-mega boards, #define buffSize 128. May need to increase.
    b: Uncomment #define ENABLE_RECORDING and #define BLOCK_COUNT 10000UL

2. Usage is as below. See https://github.com/TMRh20/TMRpcm/wiki/Advanced-Features#wiki-recording-audio for
   additional informaiton.
*/


#define SD_ChipSelectPin 10
TMRpcm audio;
int audiofile = 0;
unsigned long i = 0;
bool recmode = 0;

void setup() {
  Serial.begin(9600);
  pinMode(A0, INPUT);
  pinMode(PIN_GATE_IN, INPUT_PULLUP);
  attachInterrupt(0, button, LOW);
  SD.begin(SD_ChipSelectPin);
  audio.CSPin = SD_ChipSelectPin;
  Serial.println("Initialized");

}

void loop() {
}

void button() {
  while (i < 300000) {
    i++;
  }
  i = 0;
  if (recmode == 0) {
    recmode = 1;
    audiofile++;
    Serial.println("Started");
    switch (audiofile) {
      case 1: audio.startRecording("1.wav", 16000, A0); break;
      case 2: audio.startRecording("2.wav", 16000, A0); break;
      case 3: audio.startRecording("3.wav", 16000, A0); break;
      case 4: audio.startRecording("4.wav", 16000, A0); break;
      case 5: audio.startRecording("5.wav", 16000, A0); break;
      case 6: audio.startRecording("6.wav", 16000, A0); break;
      case 7: audio.startRecording("7.wav", 16000, A0); break;
      case 8: audio.startRecording("8.wav", 16000, A0); break;
      case 9: audio.startRecording("9.wav", 16000, A0); break;
    }
  }
  else {
    recmode = 0;
    Serial.println("stopped");
    switch (audiofile) {
      case 1: audio.stopRecording("1.wav"); break;
      case 2: audio.stopRecording("2.wav"); break;
      case 3: audio.stopRecording("3.wav"); break;
      case 4: audio.stopRecording("4.wav"); break;
      case 5: audio.stopRecording("5.wav"); break;
      case 6: audio.stopRecording("6.wav"); break;
      case 7: audio.stopRecording("7.wav"); break;
      case 8: audio.stopRecording("8.wav"); break;
      case 9: audio.stopRecording("9.wav"); break;
    }
  }
}

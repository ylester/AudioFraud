# import library

import RPi.GPIO as GPIO

# to use Raspberry Pi board pin numbers

GPIO.setmode(GPIO.BOARD) # or GPIO.setmode(GPIO.BCM)

# set up the GPIO Pins – for input or output

GPIO.setup(11, GPIO.IN)

GPIO.setup(13, GPIO.OUT)

# taking input value from Pin 11
input_value = GPIO.input(11)

# setting output value to Pin 13
GPIO.output(13, GPIO.HIGH)

#GPIO.output(13, GPIO.LOW)
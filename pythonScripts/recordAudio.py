import subprocess                       # import library for command line interface
import time 

fileName = "rand"
fileExt = ".wav"
fileCount = 1
recordCMD1 = "arecord -D hw:1,0 -d 5 -f cd -r 48000 "
recordCMD2 = " -c 1"

while True:
    runAgain = input("Do you want to start? Enter y/n: ")
    if runAgain == "y":
        fullFileName = fileName + str(fileCount) + fileExt
        recordCMD = recordCMD1 + fullFileName + recordCMD2
        print("Will execute in command line\n"+ recordCMD)
        p = subprocess.Popen(recordCMD, shell=True, stdout=subprocess.PIPE)
        print ("recording")
        time.sleep(5)
        print ("Done rec.")
        fileCount+=1
    else:
        print("Thank you for trying our prject")
        break
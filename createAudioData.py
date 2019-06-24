from pydub import AudioSegment
import os,io


exe_directory_in_str = "/Users/sedrick/Documents/AudioFraud/AudioFraud"
data_directory_in_str = "/users/sedrick/Documents/AudioFraud/AudioFraud/data"
directory = os.fsencode(exe_directory_in_str)

def createData():
    #iterate over mp3 files
    print(os.listdir(directory))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #full numbers in the 1000s means seconds
        if filename.endswith("wav"):
            start = 0
            end = 3000
            dotindex = filename.index('.')
            dir_name = filename[:dotindex]
            file = open(file, "rb")
            sound = AudioSegment.from_wav(file)
            while end <= len(sound):
                newfile = sound[start:end]
                newfilename = dir_name + str(int((start/1000))) + "-" + str(int((end/1000))) + "seconds.wav"
                newfile.export(data_directory_in_str + '/' + dir_name + '/' + newfilename, format="wav")
                start = end
                end += 3000
            newfile = sound[end:]
            newfilename = dir_name + "remainingseconds.wav"
            newfile.export(data_directory_in_str + '/' + dir_name + '/' + newfilename,
                           format='wav')
            file.close()
        else:
         continue

createData()
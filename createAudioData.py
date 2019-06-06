from pydub import AudioSegment
import os


directory_in_str = "/Users/sedrick/Documents/AudioFraud/AudioFraud/data"
directory = os.fsencode(directory_in_str)

def createData():
    #iterate over mp3 files
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #full numbers in the 1000s means seconds
        if filename.endswith("mp3"):
            start = 0
            end = 3000
            dotindex = filename.index('.')
            dir_name = filename[:dotindex]
            sound = AudioSegment.from_mp3(file)
            while end <= len(sound):
                newfile = sound[start:end]
                newfilename = filename + (start/1000) + "-" + (end/1000) + "seconds"
                newfile.export('/Users/sedrick/Documents/AudioFraud/data/' + dir_name + '/' + newfilename + '.mp3', format="mp3")
                start = end
                end += 3000
            newfile = sound[end:]
            newfilename = filename + "remainingseconds"
            newfile.export('/Users/sedrick/Documents/AudioFraud/data/' + dir_name + '/' + newfilename + '.mp3',
                           format="mp3")
        else:
         continue

createData()
import py

def extract_features():
    None

def identify_speaker():
    None

# Test File

def main():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    us_voices = filter(lambda x: x.languages == ['en_US'], voices)
    for voice in us_voices:
        print(voice)

if __name__ == '__main__':
    main()

import speech_recognition as sr

def format_speech(list_of_sentences):
    format = []
    for sentence in list_of_sentences:
        sentence = sentence.split()

        # Append 'START' and 'END' to match formatting
        sentence = ['START'] + sentence + ['END']
        format.append(sentence)

    return format
def speech():
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    list_of_sentences = []

    # Open the microphone and start listening for speech
    with sr.Microphone() as source:
        print("Say something:")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)  # Listen for up to 5 seconds of audio

                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio)
                print("You said:", text)

                if text.lower() == "stop listening":
                    print("Stopping the script.")
                    break  # Exit the loop when "stop" is detected
                else:
                    list_of_sentences.append(text)
                    print("Say something again:")

            except sr.WaitTimeoutError:
                print("No speech detected within the timeout.")

            except sr.UnknownValueError:
                print("Could not understand audio.")
    return list_of_sentences

if __name__ == '__main__':
    text = speech()
    print(format_speech(text))
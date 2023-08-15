import speech_recognition as sr
import tkinter as tk
import algorithm

sentences = []
text_display = None

def update_text(): 
    global recognized_text
    text_display.config(state=tk.NORMAL)
    text_display.insert(tk.END, recognized_text + '\n')
    text_display.config(state=tk.DISABLED)
    text_display.see(tk.END)

def format_speech(list_of_sentences):
    format = []
    for sentence in list_of_sentences:
        sentence = sentence.split()

        # Append 'START' and 'END' to match formatting
        sentence = ['START'] + sentence + ['.'] + ['END']
        format.append(sentence)

    return format
def speech(training_data):
    global recognized_text, prev_text, sentences

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Open the microphone and start listening for speech
    with sr.Microphone() as source:

        while True:
            try:
                audio = recognizer.listen(source, timeout=None)
                recognized_text = recognizer.recognize_google(audio)

                if recognized_text.lower() == "stop listening":
                    recognized_text = "Stopped listening."
                    update_text()
                    break  # Exit the loop when "stop" is detected

                elif recognized_text.lower() == "translate all":
                    recognized_text = str(algorithm.viterbi(training_data, format_speech(sentences)))

                elif recognized_text.lower() == "translate":
                    recognized_text = str(algorithm.viterbi(training_data, format_speech([prev_text])))

                elif recognized_text.lower() == "clear all":
                    recognized_text = "Cleared."
                    sentences.clear()
                else:
                    sentences.append(recognized_text)
                    prev_text = recognized_text

                update_text()

            except sr.WaitTimeoutError:
                pass

            except sr.UnknownValueError:
                recognized_text = "Could not understand audio."
                update_text()

def application(training_data):
    global text_display

    root = tk.Tk()
    root.title("Speech Recognition GUI")

    # Create a text widget to display recognized text
    text_display = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
    text_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Create a thread to listen for speech
    import threading
    speech_thread = threading.Thread(target=speech, args=(training_data,))
    speech_thread.start()

    root.mainloop()
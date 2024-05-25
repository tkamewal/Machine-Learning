import sys
print(sys.executable)
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.record(source,duration=5)
    query1=r.recognize_google(audio)
    query1=query1.lower()
    print(query1)

import sounddevice
from scipy.io.wavfile import write
fs = 44100
second = 5

print("Recording....")
record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
sounddevice.wait()
write("output.wav", fs, record_voice)
print("Finished recording")

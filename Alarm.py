import datetime
import os

# Get the current date and time
date = datetime.datetime.now()
print(date)
# Input alarm date and time
# D = int(input("Enter the date: "))
# M = int(input("Enter the month: "))
# Y = int(input("Enter the year: "))
# H = int(input("Enter the hour: "))
# m = int(input("Enter the minute: "))
# s = int(input("Enter the second: "))

Y, M, D = input("Enter the Date(YYYY/MM/DD): ").split("/")
H, m, s = input("Enter the Time(HH:MM:SS): ").split(":")

print(f"The Alarm Set for {H}:{m}:{s} on {D}/{M}/{Y}")

while True:
    # Update the current date and time inside the loop
    date = datetime.datetime.now()

    # Check if the current time has reached or surpassed the set alarm time
    if date.day == int(D) and date.month == int(M) and date.year == int(Y) and date.hour == int(H) and date.minute == int(m) and date.second == int(s):
        print("Wake Up!")
        os.startfile(r"C:\Users\TANMAY KAMEWAL\OneDrive\Desktop\VoiceDetection\Elon1.mp3")
        break

import cv2
cap = cv2.VideoCapture("C:\\Users\\TANMAY KAMEWAL\\Downloads\\SampleVideo_1280x720_1mb.mp4")
total = cap.get(7)
print(total)

cap.set(1,78)
ret,frame = cap.read()
cv2.imwrite('h'+'.jpg',frame)

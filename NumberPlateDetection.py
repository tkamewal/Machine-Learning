import torch as pt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr

image = cv2.imread("C:\\Users\\TANMAY KAMEWAL\\Downloads\\pexels-kelvin-valerio-1519192.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image directly
plt.imshow(gray, cmap='gray')
plt.show()
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(edged, cmap='gray')
plt.show()

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(new_image)
plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(image, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE
                  )
res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
plt.imshow(res)
plt.show()

import numpy as np
import cv2
import pyautogui as image

fourcc  = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('output.avi',fourcc,60.0,(1366,768))

while True:
    img = image.screenshot()
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    cv2.imshow('Recording Window',frame)
    out.write(frame)

    # cv2.imshow('Screen',frame)
    if cv2.waitKey(1) == 27:
        break
out.release()
cv2.destroyAllWindows()

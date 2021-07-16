import cv2

font = cv2.FONT_HERSHEY_DUPLEX
text = 'Welcome illi'
textsize = cv2.getTextSize(text, font, 1, 2)
print(cv2.getTextSize(text, font, 1, 2)[0][0])
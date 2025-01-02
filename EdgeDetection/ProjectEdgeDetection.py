import numpy as np
import cv2
import imutils
import pytesseract
from skimage.segmentation import clear_border

def show(s, i):
    cv2.imshow(s, i)
    cv2.waitKey(0)

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def recognize_license_plate(frame):
    obj = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    candidates = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = imutils.grab_contours(candidates)
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)[:15]

    lpText = None
    lpContour = None
    reg = None

    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        aspRatio = w / float(h)
        
        if w >= 1 and aspRatio <= 4:
            lpContour = c
            lp = gray[y:y+h, x:x+w]
            reg = cv2.threshold(lp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            reg = clear_border(reg)
            break

    if reg is not None:
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(7)
        lpText = pytesseract.image_to_string(reg, config=options)

    if lpText is not None and lpContour is not None:
        lpText = cleanup_text(lpText)
        return lpText
    return None

def process_toll_payment(lpText, bank_accounts, toll_fee):
    if lpText in bank_accounts:
        balance = bank_accounts[lpText]
        if balance >= toll_fee:
            bank_accounts[lpText] -= toll_fee
            print(f"Toll fee of {toll_fee} deducted for {lpText}. Remaining balance: {bank_accounts[lpText]}")
        else:
            print(f"Insufficient funds for {lpText}. Current balance: {balance}")
    else:
        print(f"Plate {lpText} not found in bank accounts.")

if __name__ == "__main__":
    bank_accounts = {
        "E2101PAD": 100000,
        "B2540BFA": 50000,
        "T1192EV": 25000,
        "BG1632RA": 4000,
        "B1716SDC": 80000,
        "AD8693AS": 9000,
        "B6703WJF": 780000,
        "B1770SCY": 82000,
        "D1006QZZ": 20000,
        "B7716SDC": 0        
    }
    toll_fee = 16000

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        exit()

    detected_plates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        lpText = recognize_license_plate(frame)
        if lpText:
            print(f"Detected Plate: {lpText}")
            detected_plates[lpText] = detected_plates.get(lpText, 0) + 1
        
        cv2.imshow("Toll Booth", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if detected_plates:
        most_frequent_plate = max(detected_plates, key=detected_plates.get)
        print(f"Most Frequent Plate: {most_frequent_plate}")
        process_toll_payment(most_frequent_plate, bank_accounts, toll_fee)
    else:
        print("No plates detected.")

    cap.release()
    cv2.destroyAllWindows()
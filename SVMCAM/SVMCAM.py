import cv2
import numpy as np
import easyocr
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from joblib import dump, load
import time

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    features, _ = hog(
        gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )
    return features

def gocr(img, bbox):
    x, y, w, h = bbox
    roi = img[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)

    reader = easyocr.Reader(["en"], gpu=True)
    result = reader.readtext(enhanced)
    return result

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

def main():
    svm = load("svm_license_plate_model.joblib")
    scaler = load("scaler.joblib")

    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

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

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Webcam unavailable!")
        return

    print("Starting webcam feed. Press 'Esc' to exit.")

    fps_limit = 10
    prev_time = 0
    last_ocr_time = 0
    ocr_interval = 3

    while True:
        time_elapsed = time.time() - prev_time
        if time_elapsed < 1.0 / fps_limit:
            continue

        ret, frame = video.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        prev_time = time.time()
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20))

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = frame[y:y+h, x:x+w]

            try:
                roi_resized = cv2.resize(roi, (128, 128))
                gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY) if len(roi_resized.shape) == 3 else roi_resized
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray_roi)

                features = extract_hog_features(enhanced)
                features = scaler.transform([features])
                prediction = svm.predict(features)

                if prediction[0] == 1:
                    current_time = time.time()
                    if current_time - last_ocr_time > ocr_interval:
                        detected_text = gocr(frame, (x, y, w, h))
                        if detected_text:
                            lpText = detected_text[0][-2].replace(" ", "")
                            print(f"Detected Plate: {lpText}")
                            process_toll_payment(lpText, bank_accounts, toll_fee)
                        last_ocr_time = current_time

            except Exception as e:
                print(f"Error during prediction: {e}")

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

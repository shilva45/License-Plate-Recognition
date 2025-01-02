from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
import re

def gocr(img, d):
    x = int(d[0])
    y = int(d[1])
    w = int(d[2])
    h = int(d[3])

    img = img[y:h, x:w]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = reader.readtext(gray)

    max_area = 0
    best_text = ""

    for r in res:
        bbox, text, confidence = r
        if confidence > 0.2:
            (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                best_text = text

    best_text = "".join(best_text.split()).upper()

    best_text = re.sub(r"[^A-Z0-9]", "", best_text)

    if best_text.startswith("I"):
        best_text = best_text[1:]

    match = re.match(r"^([A-Z]{1,2}\d{1,4}[A-Z]{1,3})", best_text)
    if match:
        best_text = match.group(1)

    return best_text

if __name__ == "__main__":
    car_model = YOLO("best.pt")
    reader = easyocr.Reader(["en"])

    video_source = 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        exit()

    detected_texts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot fetch the frame.")
            break

        detections = car_model(frame)[0]

        for d in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            detected_text = gocr(frame, [x1, y1, x2, y2])
            print(f"Detected: {detected_text}")

            if re.match(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$", detected_text):
                detected_texts[detected_text] = detected_texts.get(detected_text, 0) + 1

            cv2.putText(frame, detected_text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(detections.boxes) == 0:
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            detected_text = gocr(frame, [x1, y1, x2, y2])
            print(f"No detections, OCR result: {detected_text}")

            if re.match(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$", detected_text):
                detected_texts[detected_text] = detected_texts.get(detected_text, 0) + 1

            cv2.putText(frame, detected_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video Output", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):   
            detected_texts.clear()
            print("Dictionary cleared.")

    cap.release()
    cv2.destroyAllWindows()
    
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

    if detected_texts:
        most_frequent_plate = max(detected_texts, key=detected_texts.get)
        print(f"Most Frequent Plate: {most_frequent_plate}")

        if most_frequent_plate in bank_accounts:
            balance = bank_accounts[most_frequent_plate]
            if balance >= toll_fee:
                bank_accounts[most_frequent_plate] -= toll_fee
                print(f"Toll fee of {toll_fee} deducted. Remaining balance: {bank_accounts[most_frequent_plate]}")
            else:
                print(f"Insufficient funds for plate {most_frequent_plate}. Current balance: {balance}")
        else:
            print(f"Plate {most_frequent_plate} not found in bank accounts.")
    else:
        print("No plates detected.")
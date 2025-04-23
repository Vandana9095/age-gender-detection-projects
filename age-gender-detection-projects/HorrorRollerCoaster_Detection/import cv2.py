import cv2
import time
import csv
import os
from datetime import datetime

# Absolute paths to model files
age_proto = "/Users/vandanakashyap/Downloads/horror_age_detection/age_deploy.prototxt"
age_model = "/Users/vandanakashyap/Downloads/horror_age_detection/age_net.caffemodel"
gender_proto = "/Users/vandanakashyap/Downloads/horror_age_detection/gender_deploy.prototxt.txt"  # Update with the correct file path
gender_model = "/Users/vandanakashyap/Downloads/horror_age_detection/gender_net.caffemodel"

# Load models with error handling
try:
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    print("Age model loaded successfully.")
except Exception as e:
    print("Age model failed to load:", e)

try:
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
    print("Gender model loaded successfully.")
except Exception as e:
    print("Gender model failed to load:", e)

# Mean values and label ranges
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize video capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# CSV logging setup
log_entries = []

print("Starting camera... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w < 30 or h < 30:
            continue  # Skip very small faces

        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
            continue  # Skip empty faces

        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Estimate approximate age
            age_range = age.replace('(', '').replace(')', '').split('-')
            age_low, age_high = int(age_range[0]), int(age_range[1])
            approx_age = (age_low + age_high) // 2

            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            if approx_age < 13 or approx_age > 60:
                # Red rectangle and "Not allowed"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Not allowed", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_entries.append([approx_age, gender, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            else:
                # Green rectangle and "Allowed"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Allowed", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print("Error processing face:", e)

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save senior citizen and child entries to CSV
if log_entries:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'/Users/vandanakashyap/Downloads/senior_citizen_log_{timestamp}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Age', 'Gender', 'Time'])
        writer.writerows(log_entries)
    print(f"Logged disallowed entries to CSV: {filename}")
else:
    print("No disallowed entries to log.")

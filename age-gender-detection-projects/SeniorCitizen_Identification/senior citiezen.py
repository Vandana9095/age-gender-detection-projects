import cv2
import pandas as pd
import datetime
from deepface import DeepFace

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize CSV file
csv_file = "senior_citizen_visits.csv"
data_columns = ["Age", "Gender", "Time"]

# Create DataFrame and save if not exists
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=data_columns)
    df.to_csv(csv_file, index=False)

def detect_age_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Predict age and gender using DeepFace
            analysis = DeepFace.analyze(face_img, actions=["age", "gender"], enforce_detection=False)
            age = analysis[0]['age']
            gender = analysis[0]['dominant_gender']
            
            # Mark as senior citizen if age > 60
            label = f"{gender}, {age}"
            if age > 60:
                label += " (Senior)"
                log_visit(age, gender)
            
            # Draw rectangle & label on detected face
            color = (0, 255, 0) if age > 60 else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        except Exception as e:
            print("Error in prediction:", str(e))
    
    return frame

def log_visit(age, gender):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[age, gender, timestamp]], columns=data_columns)
    new_entry.to_csv(csv_file, mode='a', header=False, index=False)
    print(f"Logged: Age={age}, Gender={gender}, Time={timestamp}")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = detect_age_gender(frame)
    cv2.imshow('Senior Citizen Identification', processed_frame)
    print("Processing frame...")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

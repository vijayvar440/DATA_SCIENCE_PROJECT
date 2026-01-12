# import cv2
# ural = "http://192.168.31.221:8080/video"

# face_cascade = cv2.CascadeClassifier("OpenCV\phase_6\haarcascade_frontalcatface_extended.xml")
# eye_cascade = cv2.CascadeClassifier("OpenCV\phase_6\haarcascade_eye.xml")
# smile_cascade = cv2.CascadeClassifier("OpenCV\phase_6\haarcascade_smile.xml")

# cap = cv2.VideoCapture(ural)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray,1.1,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w, y+h) (y+w, y+h), (0,255,0), 2)

#     roi_gray = gray[y:y+h, x:x+w]   

#     roi_colour = frame[y:y+h, x:x+w]    

#     eys = eye_cascade.detectMultiScale(roi_gray, 1.1 , 10 )
#     if len(eys) > 0:
#         cv2.putText(frame,"Eyes Detection",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX)
    
#     smil = smile_cascade.detectMultiScale(roi_gray, 1.7 , 20 )
#     if len(eys) > 0:
#         cv2.putText(frame,"Smil  Detection",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX)
    
#     cv2.imshow("Smart face detaction ",frame)
#     if cv2.waitKey(1)& 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()
  

import cv2

ural = "http://192.168.31.221:8080/video"
import cv2

face_cascade = cv2.CascadeClassifier(
    r"OpenCV\phase_6\haarcascade_smile.xml"
)
eye_cascade = cv2.CascadeClassifier(
    r"OpenCV\phase_6\haarcascade_eye.xml"
)
smile_cascade = cv2.CascadeClassifier(
    r"OpenCV\phase_6\haarcascade_smile.xml"
)

# Safety check
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Error loading cascade files")
    exit()

cap = cv2.VideoCapture(ural)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        if len(smile) > 0:
            cv2.putText(frame, "Smile Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Face Eye Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import face_recognition
import cv2
import numpy as np

video_capture=cv2.VideoCapture(0)
print(video_capture)

sandip_fachara_image = face_recognition.load_image_file("sandip1.jpg")
sandip_fachara_face_encoding = face_recognition.face_encodings(sandip_fachara_image)[0]

known_face_encodings = [
    sandip_fachara_face_encoding,
]

known_face_name = [
    "Sandip"
]

face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

while  True:
    #video_capture.open(0)
    ret, frame = video_capture.read()
    #print(video_capture.isOpened())
    #print(ret)
    #print(frame)
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    reb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(reb_small_frame)
        face_encodings = face_recognition.face_encodings(reb_small_frame, face_locations)
        face_name = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index= np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_name[best_match_index]
            face_name.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)
        
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("================================")
video_capture.release()
cv2.destroyALLWindows()



'''requirments to run facere

1.pip install opencv-python
2.HTTP//pypu.org/simple/did
'''

        



    

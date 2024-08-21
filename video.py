import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
video = cv2.VideoCapture(0)  # Adjust the index if necessary
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # Change to "hog" if you prefer the HOG model

print("Loading known faces...")

known_faces = []
known_names = []

# Load known faces and names
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_faces.append(encoding)
            known_names.append(name)

print("Known faces loaded. Processing unknown faces...")

while True:
    ret, image = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        # Find all face locations and encodings in the current frame
        locations = face_recognition.face_locations(image_rgb, model=MODEL)
        encodings = face_recognition.face_encodings(image_rgb, locations)
    except Exception as e:
        print(f"Error during face recognition: {e}")
        continue

    # Process each detected face
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match Found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]

            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Display the resulting image
    cv2.imshow('Face Recognition', image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

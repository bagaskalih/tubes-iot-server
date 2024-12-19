import os
import face_recognition
import pickle

# Path to the dataset directory
DATASET_DIR = "dataset"

# load registered faces
known_encodings = []
known_names = []

try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
except FileNotFoundError:
    print("No encodings found. Starting fresh.")

# Load a sample picture and learn how to recognize it.
for file_path in os.listdir(DATASET_DIR):
    name = file_path.split(".")[0]
    if name in known_names:
        print("Already registered:", name, ". Skipping...")
        continue

    file_path = os.path.join(DATASET_DIR, file_path)
    image = face_recognition.load_image_file(file_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    print("Registered:", name)
    known_names.append(name)

# store the encodings and names
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print("Encodings saved to encodings.pkl")
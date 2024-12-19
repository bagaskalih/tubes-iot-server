import cv2
import face_recognition
import pickle
import numpy as np
from flask import Flask, request, render_template
import threading
import time
from supabase import create_client

# Flask app initialization
app = Flask(__name__)

# Load encodings and names from the pickle file
try:

    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
except FileNotFoundError:
    print("No encodings found. Starting fresh.")
    print("Please run learn_faces.py to register faces.")
    exit(1)

# Global variables
camera_idx = 0 # Camera index
frame_count = 0 # Frame counter
video_capture = None # Video capture object
camera_running = False # Camera status
last_motion_time = None # Time of last motion detection
motion_timeout = 5  # Timeout duration in seconds
supabase_url = "https://sbbybeiuhpupwdsytcpx.supabase.co" # Supabase URL
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNiYnliZWl1aHB1cHdkc3l0Y3B4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ1NjM2NTAsImV4cCI6MjA1MDEzOTY1MH0.yoIIcexQLnM78olPY58HWRLRvhG8_EYcGiPCyrfMZYs"
supabase = create_client(supabase_url, supabase_key) # instantiate Supabase client
table_name = 'test2' # Table name


def insert_data(nama, nim):
    response = supabase.table(table_name).insert([{'nama': nama, 'nim': nim}]).execute()
    return response

def start_camera(reduced=False):
    """Face recognition processing with motion timeout."""
    global frame_count, video_capture, camera_running, last_motion_time

    if camera_running:
        return  # Prevent multiple instances of the camera
    camera_running = True

    window_name = "Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        if reduced:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the motion timeout timer
        if last_motion_time and time.time() - last_motion_time > motion_timeout:
            print("No motion detected for 5 seconds. Stopping camera.")
            break  # Stop the camera if no motion detected for 5 seconds

        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if frame_count % 5 == 0:  # Process every 5th frame
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        frame_count += 1

        # Match detected faces with known faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"
            if face_distances[best_match_index] < 0.6:  # Threshold for recognition
                name = known_names[best_match_index]
                nama = name.split("_")[0]
                nim = name.split("_")[-1]
                
                # only insert data if the name is not in the database
                response = supabase.table(table_name).select('*').eq('nim', nim).execute()

                if len(response.data) == 0:
                    insert_data(nama, nim)
                    print(f"Data {nama} inserted to database")
                else:
                    print(f"Data {nama} already exists in database")

            # Scale back up face locations to original frame size
            if reduced:
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({face_distances[best_match_index]:.2f})",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    frame_count = 0
    video_capture.release()
    cv2.destroyWindow(window_name)
    camera_running = False


@app.route('/motion', methods=['POST'])
def motion_detected():
    global video_capture, last_motion_time

    data = request.get_json()
    if data.get("motion", False):
        if video_capture is None or not video_capture.isOpened():
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        last_motion_time = time.time()

        if not camera_running:
            threading.Thread(target=start_camera, args=(False,)).start()

    return "Motion received, starting camera", 200


@app.route('/data', methods=['GET'])
def get_data():
    """Fetch data from Supabase."""
    response = supabase.table(table_name).select('*').execute()
    data = response.data
    return {"data": data}, 200


@app.route('/view_data', methods=['GET'])
def view_data():
    response = supabase.table(table_name).select('*').execute()
    data = response.data
    return render_template('index.html', data=data)


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)

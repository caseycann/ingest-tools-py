import cv2
import os

def save_spread_out_faces(video_path, output_folder, max_detections=10):
    # Extract the base filename without the extension
    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Calculate the total number of frames, frame rate, and the interval for detections
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detection_interval = max(1, total_frames // max_detections)

    detection_count = 0
    frame_count = 0

    while detection_count < max_detections:
        # Set the next frame to be read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Read the frame
        ret, frame = cap.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Check if any faces are detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw a rectangle around each face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            detection_count += 1

            # Calculate the timecode (HHMMSSFF)
            hours, rem = divmod(frame_count / fps, 3600)
            minutes, seconds = divmod(rem, 60)
            frame_within_second = int(frame_count % fps)
            timecode = f"{int(hours):02}{int(minutes):02}{int(seconds):02}{frame_within_second:02}"

            # Save the frame with highlighted faces
            frame_filename = f"{output_folder}/{base_filename}_{timecode}.jpg"
            cv2.imwrite(frame_filename, frame)

        # Move to the next detection point
        frame_count += detection_interval

    # Release the video capture object
    cap.release()

# Example usage
save_spread_out_faces("/Users/cac2338/Desktop/__samSmith-test.mp4", "/Users/cac2338/Desktop/_facial-detection")

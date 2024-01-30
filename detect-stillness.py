import cv2
import numpy as np
import os

def are_images_unique(img1, img2, threshold=20):
    """Compare two images to determine if they are unique based on a given threshold."""
    # Convert images to grayscale for comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute the absolute difference and check if it's below the threshold
    diff = cv2.absdiff(gray1, gray2)
    if np.max(diff) < threshold:
        return False  # Images are not unique
    return True  # Images are unique

def find_and_save_unique_still_frames(video_path):
    # Extract the directory where the video is located
    video_dir = os.path.dirname(video_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_idx = 0
    still_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)

        if np.max(diff) < 10:  # Adjust threshold as needed
            still_frames.append(prev_frame)

        prev_gray = gray
        prev_frame = frame
        frame_idx += 1

    cap.release()

    # Compare still frames and save only unique ones
    unique_frames = []
    for idx, frame in enumerate(still_frames):
        is_unique = True
        for unique_frame in unique_frames:
            if not are_images_unique(frame, unique_frame):
                is_unique = False
                break
        if is_unique:
            unique_frames.append(frame)
            output_path = os.path.join(video_dir, f"unique_still_frame_{len(unique_frames)-1}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
    
    cv2.destroyAllWindows()

# Replace 'path_to_your_video.mp4' with your video file path
find_and_save_unique_still_frames('/Users/cac2338/Desktop/cardDetection_test/closeupCards.mp4')

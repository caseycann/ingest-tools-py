import cv2
import numpy as np
import os

def capture_index_cards(video_path, max_images=20):
    cap = cv2.VideoCapture(video_path)
    video_dir = os.path.dirname(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Initialize the background subtractor with a slower learning rate
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    learningRate = 0.9  # Adjust as needed

    saved_images = 0
    frames_since_last_save = 0  # Counter to implement debounce mechanism
    debounce_threshold = 60  # Number of frames to wait before saving next image

    while saved_images < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fgMask = backSub.apply(frame, learningRate=learningRate)

        # Apply morphological operations to reduce noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1500:  # Adjust based on your scenario
                if frames_since_last_save >= debounce_threshold:
                    image_filename = os.path.join(video_dir, f'full_frame_{saved_images}.jpg')
                    cv2.imwrite(image_filename, frame)
                    saved_images += 1
                    frames_since_last_save = 0  # Reset the counter
                break  # Save only one image per set of contours

        frames_since_last_save += 1  # Increment the counter for debounce mechanism

        if saved_images >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()


# Replace 'path_to_your_video.mp4' with your video file path
capture_index_cards('/Users/cac2338/Desktop/cardDetection_test/_closeupCards.mp4')

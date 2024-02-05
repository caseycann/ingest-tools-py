import cv2
import numpy as np
import os

def capture_index_cards(video_path, max_images=20):
    cap = cv2.VideoCapture(video_path)
    video_dir = os.path.dirname(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    backSub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=20, detectShadows=True)
    learningRate = 0.9

    saved_images = 0
    debounce_threshold = 60
    frames_since_last_save = debounce_threshold  # Start ready to save

    # Prepare a motion history image
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    motion_history = np.zeros((h, w), dtype=np.float32)

    while saved_images < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        fgMask = backSub.apply(frame, learningRate=learningRate)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        # Update motion history
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get the timestamp in seconds
        cv2.motempl.updateMotionHistory(fgMask, motion_history, timestamp, duration=0.5)

        # Calculate the motion gradient and the motion segmentation mask
        delta1 = 0.4  # Minimum value for motion gradient magnitude
        delta2 = 1.0  # Maximum value for motion gradient magnitude
        # Directly use the motion history image without converting to uint8
        mg_mask, mg_angle = cv2.motempl.calcMotionGradient(motion_history, delta1, delta2, apertureSize=3)

        # Calculate the amount of motion
        motion_amount = cv2.norm(mg_mask, cv2.NORM_L1)

        # Check for significant motion and save the frame if there's no significant motion for a while
        if motion_amount < 1000 and frames_since_last_save >= debounce_threshold:  # Tune these thresholds
            image_filename = os.path.join(video_dir, f'full_frame_{saved_images}.jpg')
            cv2.imwrite(image_filename, frame)
            saved_images += 1
            frames_since_last_save = 0
        else:
            frames_since_last_save += 1

        if saved_images >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()



# Replace 'path_to_your_video.mp4' with your video file path
capture_index_cards('/Users/cac2338/Desktop/cardDetection_test/_closeupCards.mp4')

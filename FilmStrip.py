import cv2
from PIL import Image
import argparse
import os

# Constants for print layout
PAGE_WIDTH_INCHES = 11
PAGE_HEIGHT_INCHES = 17
STRIP_WIDTH_INCHES = 1.25
DPI = 300  # Dots per inch for printing

def extract_and_resize_frames(video_path, step=10, resize_width=300):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        for _ in range(step - 1):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        frame_height = int(frame.shape[0] * (resize_width / frame.shape[1]))
        frame = cv2.resize(frame, (resize_width, frame_height), interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()
    return frames

def create_multiple_pages(frames, strip_width_inches, page_width_inches, page_height_inches, dpi):
    strip_width_px = int(strip_width_inches * dpi)
    page_width_px = int(page_width_inches * dpi)
    page_height_px = int(page_height_inches * dpi)

    num_columns = page_width_px // strip_width_px
    frame_aspect_ratio = frames[0].shape[1] / frames[0].shape[0]
    frame_height_px = int(strip_width_px / frame_aspect_ratio)

    pages = []
    current_page = Image.new('RGB', (page_width_px, page_height_px), 'white')

    x_offset = 0
    y_offset = 0
    column = 0

    for frame in frames:
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_image = frame_image.resize((strip_width_px, frame_height_px), Image.ANTIALIAS)

        if y_offset + frame_height_px > page_height_px:
            x_offset += strip_width_px
            y_offset = 0
            column += 1
            if column >= num_columns:
                pages.append(current_page)
                current_page = Image.new('RGB', (page_width_px, page_height_px), 'white')
                x_offset = 0
                column = 0

        current_page.paste(frame_image, (x_offset, y_offset))
        y_offset += frame_height_px

    pages.append(current_page)  # Add the last page
    return pages

def process_video(video_path, strip_width_inches=1.25, page_width_inches=11, page_height_inches=17, dpi=300):
    frames = extract_and_resize_frames(video_path)
    pages = create_multiple_pages(frames, strip_width_inches, page_width_inches, page_height_inches, dpi)
    return pages

def main():
    parser = argparse.ArgumentParser(description='Process a video into a film strip.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()

    video_dir = os.path.dirname(args.video_path)
    output_dir = os.path.join(video_dir, 'film_strips')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pages = process_video(args.video_path, STRIP_WIDTH_INCHES, PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES, DPI)

    for i, page in enumerate(pages):
        output_path = os.path.join(output_dir, f'film_strip_page_{i+1}.png')
        page.save(output_path)
        print(f'Saved: {output_path}')

if __name__ == '__main__':
    main()

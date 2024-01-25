import subprocess
import sys
import json

def extract_metadata(file_path):
    # Command to use ffprobe to extract metadata in JSON format
    command = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams', 
        file_path
    ]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running ffprobe.")
        sys.exit(1)

    return json.loads(result.stdout)

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_video_metadata.py [video_file_path]")
        sys.exit(1)

    file_path = sys.argv[1]
    metadata = extract_metadata(file_path)

    # Print the metadata or save it to a file as needed
    print(json.dumps(metadata, indent=4))

if __name__ == "__main__":
    main()

from collections import Counter
import argparse
import os
import glob
import subprocess
import numpy as np
from scipy.io import wavfile
from PIL import Image
import face_alignment

# Set up argparse to accept video file paths
parser = argparse.ArgumentParser(description="Process video files for training.")
parser.add_argument("videos", nargs="+", help="Paths to video files.")
args = parser.parse_args()

# Define paths
data_root = "./data"
hdtf_root = os.path.join(data_root, "HDTF")
os.makedirs(hdtf_root, exist_ok=True)
images_dir = os.path.join(hdtf_root, "images")
landmarks_dir = os.path.join(hdtf_root, "landmarks")
audio_dir = os.path.join(hdtf_root, "audio_smooth")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(landmarks_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False, device="cuda"
)


# Function to extract audio and convert video
def process_video(video_id, video_path):
    # base_name = os.path.splitext(os.path.basename(video_path))[0]
    base_name = str(video_id)
    output_framed_path = os.path.join(images_dir, f"{base_name}_%d.jpg")
    output_audio_path = os.path.join(audio_dir, f"{base_name}.wav")

    # Convert video to 25 fps
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-r", "25", "-vf", "fps=25", "-start_number", "1", output_framed_path]
    )

    # Extract audio
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", "44100", "-ac", "2", output_audio_path]
    )

    # Read and process audio data
    rate, data = wavfile.read(output_audio_path)
    frame_size = rate // 25  # calculate frame size for 25 FPS

    # Split audio into chunks and save each as .npy
    for i in range(0, len(data), frame_size):
        chunk = data[i : i + frame_size]
        if len(chunk) == frame_size:
            np.save(
                os.path.join(audio_dir, f"{base_name}_{i//frame_size+1:d}.npy"), chunk
            )

    # Placeholder for more detailed processing, such as extracting landmarks
    for image_path in glob.glob(os.path.join(images_dir, f"{base_name}_*.jpg")):
        img_id, frame_id = os.path.basename(image_path).split(".")[0].split("_")
        output_landmark_path = os.path.join(landmarks_dir, f"{img_id}_{frame_id}.lms")
        image = Image.open(image_path)
        image_np = np.array(image)
        preds = fa.get_landmarks(image_np)
        if preds is not None:
            np.savetxt(output_landmark_path, preds[0], fmt="%1.5f")

    return base_name


# Search for all .webm files and process them
file_list = sorted([f for f in args.videos])
data_entries = []
for file_id, file_name in enumerate(file_list, start=1):
    data_entries.append(process_video(file_id, file_name))

# Create training and testing text files
all_file_path = os.path.join(hdtf_root, "data.txt")
train_file_path = os.path.join(data_root, "data_train.txt")
test_file_path = os.path.join(data_root, "data_test.txt")

with open(all_file_path, "w") as f:
    for img_id, frame_cnt in sorted(Counter(
        os.path.basename(image_path).split(".")[0].split("_")[0]
        for image_path in glob.glob(os.path.join(images_dir, "*.jpg"))
    ).items()):
        f.write(f"{img_id} {frame_cnt}\n") # or such as, f.write(f" {frame_cnt}\n") 

with open(train_file_path, "w") as f:
    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):  # data_entries:
        f.write(f"{os.path.basename(image_path).split('.')[0]}\n")

with open(test_file_path, "w") as f:
    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        f.write(f"{os.path.basename(image_path).split('.')[0]}\n")

print("Data preprocessing complete.")

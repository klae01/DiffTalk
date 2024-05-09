from collections import Counter
import os
import glob
import subprocess
import numpy as np
from scipy.io import wavfile
import face_alignment
from PIL import Image

# Define paths
data_root = "./data"
hdtf_root = os.path.join(data_root, "HDTF")
if not os.path.exists(hdtf_root):
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
def process_video(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_framed_path = os.path.join(images_dir, f"{base_name}_%04d.jpg")
    output_audio_path = os.path.join(audio_dir, f"{base_name}.wav")

    # Convert video to 25 fps
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-r", "25", "-vf", "fps=25", output_framed_path]
    )

    # Extract audio
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", "44100", "-ac", "2", output_audio_path]
    )

    # Placeholder for landmark extraction
    # Extract landmarks (Implementation depends on available software or methods)

    # Save audio as .npy
    rate, data = wavfile.read(output_audio_path)
    np.save(os.path.join(audio_dir, f"{base_name}.npy"), data)

    # Extract landmarks for each frame
    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        img_id, frame_id = os.path.basename(image_path).split(".")[0].split("_")
        output_landmark_path = os.path.join(landmarks_dir, f"{img_id}_{frame_id}.lms")
        image = Image.open(image_path)
        image_np = np.array(image)
        preds = fa.get_landmarks(image_np)
        if preds is not None:
            np.savetxt(output_landmark_path, preds[0], fmt="%1.5f")

    return base_name


# Search for all .webm files and process them
file_list = [f for f in os.listdir(".") if f.endswith(".webm")]
data_entries = []
for file in file_list:
    data_entries.append(process_video(file))

# Create training and testing text files
all_file_path = os.path.join(hdtf_root, "data.txt")
train_file_path = os.path.join(data_root, "data_train.txt")
test_file_path = os.path.join(data_root, "data_test.txt")

with open(all_file_path, "w") as f:
    for img_id, frame_cnt in Counter(
        os.path.basename(image_path).split(".")[0].split("_")[0]
        for image_path in glob.glob(os.path.join(images_dir, "*.jpg"))
    ):
        f.write(f"{img_id}_{frame_cnt}\n")

with open(train_file_path, "w") as f:
    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):  # data_entries:
        f.write(f"{os.path.basename(image_path).split('.')[0]}\n")

with open(test_file_path, "w") as f:
    for image_path in glob.glob(os.path.join(images_dir, "*.jpg")):
        f.write(f"{os.path.basename(image_path).split('.')[0]}\n")

print("Data preprocessing complete.")

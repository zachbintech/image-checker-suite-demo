import cv2
import os
import shutil

# Initialize ORB detector
orb = cv2.ORB_create()

def get_descriptors(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return descriptors

def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return len(matches)

# Path to your images
folder_path = "../data/oldphotos"
output_folder = "similar_images"
os.makedirs(output_folder, exist_ok=True)

# Extract descriptors from all images
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
descriptors_list = [get_descriptors(p) for p in image_files]

# Create a dictionary to store groups
groups = []
visited = [False] * len(image_files)

# Group similar images
threshold_matches = 210  # adjust depending on how "strict" you want. Higher means more similar.

for i in range(len(image_files)):
    if visited[i]:
        continue
    group = [image_files[i]]
    visited[i] = True
    for j in range(i + 1, len(image_files)):
        if visited[j]:
            continue
        num_matches = match_descriptors(descriptors_list[i], descriptors_list[j])
        if num_matches > threshold_matches:
            group.append(image_files[j])
            visited[j] = True
    groups.append(group)

# Create a new directory for each group and move the images
for idx, group in enumerate(groups):
    group_dir = os.path.join(output_folder, f"group_{idx + 1}")
    os.makedirs(group_dir, exist_ok=True)
    for img_path in group:
        shutil.copy(img_path, group_dir)
        print(f"Moved {img_path} to {group_dir}")

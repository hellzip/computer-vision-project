import urllib.request
import os
import numpy as np
from PIL import Image

CLASSES = [
    "apple", "door", "mailbox", "book", "angel", "van", "pencil", "hexagon"
]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DATA_DIR = "quickdraw_images_dedup"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for category in CLASSES:
    print(f"Processing {category}...")
    
    url = f"{BASE_URL}{category.replace(' ', '%20')}.npy"
    npy_path = f"{category}.npy"
    
    try:
        urllib.request.urlretrieve(url, npy_path)
    except urllib.error.HTTPError as e:
        print(f"Skipped '{category}' - not found in dataset (HTTP {e.code})")
        continue
    
    data = np.load(npy_path)
    
    class_dir = os.path.join(DATA_DIR, category)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    for i in range(min(1000, len(data))):
        img_array = data[i].reshape(28, 28)
        img = Image.fromarray(img_array)
        img.save(os.path.join(class_dir, f"{i}.png"))
    
    # Clean up .npy file
    os.remove(npy_path)

print("Dataset ready!")
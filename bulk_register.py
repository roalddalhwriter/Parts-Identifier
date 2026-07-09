import io
import os
import requests
from pathlib import Path

# URL of your running FastAPI backend
API_URL = "http://localhost:8000/api/register"
BASE_DIR = Path("C:\\Users\\roald\\Downloads\\test_images")

# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jpg", ".png"}

def bulk_register():
    if not BASE_DIR.exists():
        print(f"❌ Error: The directory '{BASE_DIR}' does not exist.")
        return

    # Grab all subfolders inside test_images
    subfolders = [f for f in BASE_DIR.iterdir() if f.is_dir()]
    print(f"📂 Found {len(subfolders)} part directories. Starting upload process...\n")

    for part_dir in subfolders:
        part_id = part_dir.name
        
        # Collect all valid image paths in this subfolder
        image_paths = [
            p for p in part_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_paths:
            print(f"⚠️  Skipping '{part_id}': No valid images found.")
            continue

        if len(image_paths) < 3:
            print(f"⚠️  Skipping '{part_id}': Found {len(image_paths)} images, but backend requires at least 3.")
            continue

        print(f"🚀 Registering '{part_id}' with {len(image_paths)} images...")

        # Open files concurrently to stream them via multipart form-data
        file_handles = []
        files_payload = []
        try:
            for img_path in image_paths:
                fh = open(img_path, "rb")
                file_handles.append(fh)
                # Structure matches the expected list of files named 'images'
                files_payload.append(("images", (img_path.name, fh, f"image/{img_path.suffix[1:]}")))

            # Form text payload
            data_payload = {"part_id": part_id}

            # POST request hitting the backend router
            response = requests.post(API_URL, data=data_payload, files=files_payload)

            if response.status_code == 200:
                print(f"✅ Successfully registered '{part_id}'!")
            else:
                print(f"❌ Failed to register '{part_id}'. Server responded with status {response.status_code}: {response.text}")

        except Exception as e:
            print(f"💥 An unexpected error occurred while processing '{part_id}': {e}")
            
        finally:
            # Clean up and close all open system file descriptors safely
            for fh in file_handles:
                fh.close()
                
    print("\n🏁 Bulk registration task complete.")

if __name__ == "__main__":
    # Ensure requests library is installed: pip install requests
    bulk_register()
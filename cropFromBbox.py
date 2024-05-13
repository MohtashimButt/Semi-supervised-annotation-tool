import json
from PIL import Image
import os

# Function to process all JSON files in a directory
def process_json_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check if the file is a JSON file
            json_path = os.path.join(directory, filename)
            crop_image(json_path)

def crop_image(json_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Open the image
    image = Image.open(data['imagePath'][3:])
    image_name = json_path.split('\\')[-1].split(".")[0]
    # Loop through shapes and crop out the image tiles
    for shape in data['shapes']:
        label = shape['label']
        group_id = shape['group_id']
        points = shape['points']

        # Extract coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Swap coordinates if needed
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Crop the image
        print(group_id)
        cropped_image = image.crop((x1, y1, x2, y2))

        # Save the cropped image
        cropped_image.save(f"Unlabelled_Images/{image_name}_{group_id}.png")

    print(f"Cropping for {image_name} completed.")

# json_folder = "Labelme_bbox"
# process_json_files(json_folder)
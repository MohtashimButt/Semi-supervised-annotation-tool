import os
import base64
import json
import numpy as np
import cv2

def mask_to_json(mask, original_image_path):
    # Read the original image file
    with open(original_image_path, "rb") as f:
        image_data = f.read()
    
    # Encode the image data as a base64 string
    encoded_image_data = base64.b64encode(image_data).decode("utf-8")
    
    json_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": original_image_path,
        "imageData": encoded_image_data,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }
    
    # Extract polygons from the mask
    polygons = extract_polygons(mask)
    
    # Populate the JSON data with polygons
    for idx, polygon in enumerate(polygons):
        shape = {
            "label": f"carving{idx+1}",
            "points": polygon.tolist(),
            "group_id": idx+1,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }
        json_data["shapes"].append(shape)
    
    return json_data

def extract_polygons(mask):
    # Convert mask to binary
    binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract polygons from contours
    polygons = []
    for contour in contours:
        polygon = np.array(contour.squeeze().tolist())  # Ensure polygon is a numpy array
        polygons.append(polygon)
    
    return polygons

def jasonify(folder_name):
    # Create the output directory if it doesn't exist
    jsons_folder = "JSONS"
    os.makedirs(jsons_folder, exist_ok=True)

    # Iterate over mask image files in the "Labels" folder
    labels_folder = folder_name
    for mask_name in os.listdir(labels_folder):
        if mask_name.endswith(".png"):
            # Construct paths for the mask image and the original image
            mask_path = os.path.join(labels_folder, mask_name)
            original_image_name = os.path.splitext(mask_name)[0] + ".png"
            original_image_path = os.path.join("Images", original_image_name)
            
            # Read the mask image
            mask_image = cv2.imread(mask_path)
            
            # Convert mask to JSON
            json_data = mask_to_json(mask_image, original_image_path)
            
            # Save JSON data to file in the "JSONS" folder
            json_file_name = os.path.splitext(mask_name)[0] + ".json"
            with open(os.path.join(jsons_folder, json_file_name), "w") as json_file:
                json.dump(json_data, json_file)

    print("Conversion from MASKs to JSONs completed.")

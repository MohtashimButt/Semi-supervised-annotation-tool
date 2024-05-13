import os
import cv2

# Define input and output directories

def labelResize(input_folder):
    output_folder = "Images"

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            # Resize the image to 512x512
            resized_img = cv2.resize(img, (512, 512))
            
            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)

    print("Images resized and saved to 'Images' folder.")

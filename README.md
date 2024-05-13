# Semi-supervised-annotation-tool

You can download the weights (`deeplabv3_small_ubaid.pth`) from here: https://pern-my.sharepoint.com/:u:/g/personal/24100238_lums_edu_pk/EYYLPNHibGtMqXersiBJxyQBkK45BNw9zVFcejQHMMtQpA?e=b5t9QE

## Things you need to have
### Labelme
- Use `pip install labelme` to install LabelMe in your PC.


## HOW TO USE IT
- Download the images from Ateeq's dataset to the `Raw_Images` folder. 
- Rename them according to the convention `site_rockNumber_IMG_DSLRnumber`. For example: `GN_006_IMG_1930.png`. If the image is not in `.png`, you must convert it to `.png`. (will upload a script for that too).
- Open LabelMe with `Raw_images` as the current working folder and `Labelme_bbox` as the Output or destination folder.
- Mark bounding boxes around the carvings and assign them proper `group_id`. For example, if you're marking first carving, it's `group_id` will be 1. If you're marking second carving, it's `group_id` will be 2 and so on
- Save the bboxes through LabelMe.
- Run the `main.py`.
After running `main.py`, you'll get following things:
- Crops (non-resized) in `Unlabelled_Images` folder.
- Crops (resized) in `Images` folder.
- Masks (resized) in `Labels` folder.
- Bounding boxes' `.json` files in `JSON` folder.
\\
Now, you may open labelme with `Images` as the current working folder and `JSONS` as the Output or destination folder to correct the labels.

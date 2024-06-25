import os
import cv2
import numpy as np

# Path to the folder containing the input images
input_folder = r"C:\Users\armel\Documents\mscproject\Codes\DenseNet\no_mask\dataset\validation\normal_cin1"

# Create a folder to save the processed images
output_folder = r"C:\Users\armel\Documents\mscproject\Codes\DenseNet\rectangular_mask\validation\normal_cin1"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
input_files = os.listdir(input_folder)

# Loop through each file in the input folder
for input_file in input_files:
    # Load the image
    image_path = os.path.join(input_folder, input_file)
    image = cv2.imread(image_path)
    # Extract the red channel
    red_channel = image[:, :, 2]

    # Apply thresholding to detect the center part of the cervix
    threshold_value = 200
    ret, thresholded_image = cv2.threshold(red_channel, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the largest contour as the central part of the cervix
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw a bounding rectangle on the original image
    bounding_box_image = image.copy()
    cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness 2

    # Generate binary mask within the bounding rectangle
    binary_mask = np.zeros_like(thresholded_image)
    cv2.rectangle(binary_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    # Generate RGB mask with region of interest highlighted
    roi_mask = np.zeros_like(image)
    cv2.rectangle(roi_mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)  # White color
    roi_image = cv2.bitwise_and(image, roi_mask)

    # Convert the binary mask to binary image
    _, binary_image = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the bounding rectangle with the largest area
    largest_area = 0
    largest_rect = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_rect = (x, y, w, h)

    if largest_rect is not None:
        x, y, w, h = largest_rect

        # Crop the roi_image using the largest bounding rectangle
        cropped_roi_image = roi_image[y:y+h, x:x+w]

        # Visualize the cropped ROI image
        #plt.figure(figsize=(5, 5))
        #plt.title('Cropped ROI Image')
        #plt.imshow(cv2.cvtColor(cropped_roi_image, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.show()
       # Save the processed image with modified filename in the output folder
        output_filename = os.path.splitext(input_file)[0] + "_threshold_mask.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, cropped_roi_image)
            
        print(f'Saved {output_filename}')
    else:
        print(f'No contour found in the binary mask for image {input_file}. Saving roi_image.')

        # Save the roi_image with modified filename in the output folder
        output_filename = os.path.splitext(input_file)[0] + "_threshold_mask.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, roi_image)
            
        print(f'Saved {output_filename}')

print('Image processing and saving complete.')

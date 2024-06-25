# Segmentation-of-cervigram-images-using-image-processing-techniques
This is an image processing technique to extract the transformation zone from cervigram images using bounding boxes (circular and rectangular). The steps used in this method are described as follows:

1- The red channel is extracted from the images, as the cervix is red in nature.
2- The central region of the cervix is identified using a threshold value of 200.
3- Contours are computed around the thresholded image, and the largest contour is selected.
4- A bounding box is then calculated around the largest contour.
5- A binary mask is generated within this rectangular bounding box.
6- The binary mask is multiplied by the original image to extract the cervix in RGB.
7- Finally, the resulting image is cropped to eliminate the maximum number of black pixels.

# Results
Rectangular bounding box
![image](https://github.com/armelsida/Segmentation-of-cervigram-images-using-image-processing-techniques/assets/115725362/406ad189-ac5c-48cb-97a6-d50e97666980)

Circular bounding box
![image](https://github.com/armelsida/Segmentation-of-cervigram-images-using-image-processing-techniques/assets/115725362/610866b7-7b1e-4d62-bafc-9da7b689cb1c)




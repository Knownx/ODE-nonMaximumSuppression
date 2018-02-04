from src.nms import nonMaximumSuppression
import numpy as np
import cv2

# Load the images along with their respective bounding boxes
images = [
	("images/audrey.jpg", np.array([
	(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)])),
	("images/bksomels.jpg", np.array([
	(114, 60, 178, 124),
	(120, 60, 184, 124),
	(114, 66, 178, 130)])),
	("images/gpripe.jpg", np.array([
	(12, 30, 76, 94),
	(12, 36, 76, 100),
	(72, 36, 200, 164),
	(84, 48, 212, 176)])),
    ("images/wxy_g.jpg", np.array([
    (80, 65, 171, 153),
    (97, 70, 169, 140),
    (82, 64, 165, 145),
    (97, 62, 191, 162)]))]

# Loop over the images
for (imagePath, boundingBoxes) in images:
    # Load the image and clone it
    print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    image = cv2.imread(imagePath)
    orig = image.copy()

    # Loop over the bounding boxes for each image and draw them
    for(startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0,0,255), 2)

    # Call the suppression procedure on the bounding boxes
    pick = nonMaximumSuppression(boundingBoxes, 0.3)
    print ("[x] after applying nonMaximumSuppression, %d bounding boxes" % (len(pick)))

    # Loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0),2)

    # Display the images
    cv2.imshow("Original", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
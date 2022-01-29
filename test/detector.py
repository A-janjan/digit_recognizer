from tensorflow.keras.models import load_model
import numpy as np
import imutils
from imutils.contours import sort_contours
import cv2
import argparse
from blur_detector import detect_blur_fft


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")    
args = vars(ap.parse_args())

print("[INFO] Loading OCR Model ...")
model = load_model("digit_recognizer.h5")

#load image and pre-process it and find character contours
input_image = args["image"]
image = cv2.imread(input_image)
if image.shape[2]==3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

(mean, blurry) = detect_blur_fft(gray, size=60)

if blurry:
    print("mean is: ", mean)
    print("your picture is blurry, so please give a better picture")
    quit()

blurred = cv2.GaussianBlur(gray, (5,5), 0)  # Gaussian blurring to reduce noise

# edge detection and find contours in edge map
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method = "left-to-right")[0]

# initialize the list of contour bounding boxes and associated 
# characters that we'll be OCR'ing
chars = []

# loop over the contours
for c in cnts:
    # compute the bounding rectangle of the contour
    (x, y, w, h) = cv2.boundingRect(c) 
    # filter out bounding rectangle, ensuring they are neither too small
    # nor too large
    if (w >= 5 ) and (h >= 15):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]    #region of interest(roi)
        #cleaning up the images using Otsu’s thresholding technique
        #goal: segmenting the characters such that they appear as white on a black background
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        # if the width is greater than the height, resize along the
        # width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)

        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=32)
        
        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        
        # pad the image and force 28x28 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded = cv2.resize(padded, (28, 28)) #resize the image to 32 x 32 pixels to round out the pre-processing phase

        # prepare the padded image for classification via our handwriting OCR model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        # update our list of characters that will be OCR'd
        chars.append((padded, (x, y, w, h)))

# extract the bounding box(rectangle) locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)

# define the list of label names

labelNames = "0123456789"
labelNames = [l for l in labelNames]

# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]

    # draw the prediction on the image
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
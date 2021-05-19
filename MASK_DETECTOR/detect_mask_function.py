from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

def set_up():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    
    net = cv2.dnn.readNet(protoPath, weightsPath)

def get_mask_prob(detections, index, image):
        
        confidence = detections[0, 0, index, 2]
        model = load_model("mask_detector.model")
        (h, w) = image.shape[:2]
        
        confThres = 0.5

        if confidence > confThres:
                box = detections[0, 0, index, 3:7] * np.array([w, h, w, h])
                # define detected face area
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                
                # define face
		        # extract the face ROI, convert it from BGR to RGB channel
		        # ordering, resize it to 224x224, and preprocess it
                
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

        		# mask check
                (mask, withoutMask) = model.predict(face)[0]
                isMask = mask > withoutMask
                return startX, startY, endX, endY, mask, withoutMask
        return -1, -1, -1, -1, -1, -1

def mask_image(image):
	(h, w) = image.shape[:2]

	net.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0)))
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):

                (startX, startY, endX, endY, mask, withoutMask) = get_mask_prob(detections, i, image)
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                
                cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return image

def display_image(image):
    cv2.imshow("Output", image)
	cv2.waitKey(0)
	
if __name__ == "__main__":
    set_up()
    image = cv2.imread("images/pic1.jpeg")
	image = mask_image(image)
    display_image(image)

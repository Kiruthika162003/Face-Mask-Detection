import cv2
import numpy as np

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face mask detection model
mask_net = cv2.dnn.readNetFromTensorflow('mask_detection_model.pb')

# Load class labels
class_labels = {0: 'Mask', 1: 'No Mask'}

# Load image
image = cv2.imread('test_image.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over detected faces
for (x, y, w, h) in faces:
    # Preprocess the face region
    face_roi = image[y:y + h, x:x + w]
    face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Pass the face blob through the mask detection model
    mask_net.setInput(face_blob)
    detections = mask_net.forward()

    # Get the predicted class label and confidence
    class_id = np.argmax(detections[0])
    confidence = detections[0][class_id]

    # Draw bounding box and label
    label = f'{class_labels[class_id]}: {confidence:.2f}'
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Display the output image
cv2.imshow('Mask Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

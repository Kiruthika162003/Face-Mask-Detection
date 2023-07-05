# Face Mask Detection

The Face Mask Detection project aims to detect whether a person is wearing a face mask or not using computer vision techniques.

## Project Explanation

The face mask detection project involves the following steps:

1. Loading Models: The project loads two pre-trained models. First, the Haar cascade face detection model is used to detect faces in an image. Then, a face mask detection model, trained using deep learning techniques, is loaded to classify each detected face as wearing a mask or not.

2. Face Detection: The loaded face detection model is applied to the input image, identifying the regions where faces are present.

3. Face Mask Detection: For each detected face, a region of interest (ROI) is extracted, preprocessed, and passed through the face mask detection model. The model predicts whether the face is wearing a mask and provides a confidence score.

4. Visualization: Bounding boxes are drawn around the detected faces, and labels indicating the presence or absence of a mask, along with confidence scores, are added to the image. The output image is then displayed.

## Requirements

- Python 3
- OpenCV
- NumPy


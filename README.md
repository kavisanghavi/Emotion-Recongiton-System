# Emotion Recongiton System
A deep learning based system to determine the emotion of a human looking at the face.
It can classify Human facial expressions into 7 basic emotions: **happy, sad, fear, anger, disgust and neutral.**
## Dataset
We used the dataset from a Kaggle Facial Expression Recognition Challenge to train the
model (**FER2013**). It was created by **Aaron Courville** and **Pierre Luc Carrier**. It was not
created by manual selection but by using the Google API by searching for all the images of
faces which could relate to 184 words related to emotions like “enganged”, “happy” etc.
Alongwith these 184 words, gender ethinicty etc were also added and in the end the output
was a set of about 650 queries for searching. The top 1000 images that were returned by each
query were stored. After that using OpenCV’s face detection method, the images with proper
faces were selected, manually checking the labels and in the end converting them to a 48 x 48
size image. The dataset comprises a total of 35887 pre-cropped, 48-by-48 pixel gray scale
images of faces each labeled with one of the 7 emotion classes: anger, disgust, fear,
happiness, sadness, surprise and neutral.

## Accuracy
The best accuracy achieved in this dataset is around **68%**. Even
human classification on this dataset done by ICML 2013 recorded that human
accuracy was also **65 + 5%**. We have been successful to achieve an accuracy of
**61%**.



# Real-Time-Object-Detection-In-Videos

## Introduction

This project aims to do real-time object detection through a laptop camera or webcam using OpenCV and MobileNetSSD. The idea is to loop over each frame of the video stream, detect objects like person, chair, dog, etc. and bound each detection in a box.
Here, we will go through the steps required for setting up the project and some explanation about the code.



### How to run this code?

**Step 1:** Create a directory in your local machine and cd into it

```
mkdir ~/Desktop/opencv_project
cd ~/Desktop/opencv_project
```

**Step 2:** Copy the project source file  and cd into the folder:

```
cd Real-Time-Object-Detection-With-OpenCV
```
**Step 3:** Install all the necessary libraries. I used MacOS for this project. These are some of the libraries I had to install:

```

pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install matplotlib
pip install imutils
pip install tkinter

```

Make sure to download and install opencv and and opencv-contrib releases for OpenCV 3.3. This ensures that the convolutional neural network (cnn) module is installed. You must have OpenCV 3.3 (or newer) to run this code.

**Step 4:** Make sure you have your video devices connected (e.g. Webcam, FaceTime HD Camera, etc.). You can list them by typing this in your terminal

```
system_profiler SPCameraDataType
system_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://"
```

**Step 5:** To start your video stream and real-time object detection, run the following command:

```
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

**Step 6:** If you need any help regarding the arguments you pass, try:

```
python real_time_object_detection.py --help
```

### References and Useful Links

* https://github.com/chuanqi305/MobileNet-SSD
* https://github.com/opencv/opencv
* https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
* https://github.com/jrosebr1/imutils

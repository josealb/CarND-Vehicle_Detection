## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./media/detections.png
[image6]: ./media/heatmap.png
[image7]: ./media/filtered.png
[image8]: ./media/video.png


[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

### #1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The first line in the first cell contains the function get_hog_features, which computes the hog features.

First, the features are computer for the entire test and train dataset. Then, they are computed frame by frame for the video.

Before computing hog, I change the colorspace to HSV, which, as we saw in the latest project is useful for disregarding changes in luminance.
Even though hog is inherently more robust to different luminances, there is still value in having similar absolute values in classification.
To choose HSV I tried different colorspaces looking for the highest validation accuracy

![alt text][image2]

### #2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided it was best to use all channels for the hog, even though with hsv you could also just use the S channel, using more channels discarded some false positives.
I kept the standard parameters for orientation, number of pixels per cell, etc.

### #3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the following features:
-Histogram of gradients
-Raw pixel data (low resolution)
-Histogram of colors

The features are then stacked and scaled and fed into the classifier for training. This is done in line 65 (svc.fit(X_train, y_train))

### Sliding Window Search

### #1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search only the windows that contain the road, from pixel 400 downwards. This reduces execution time and the probability of false positives

### #2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline is as follows.

Create windows -> Run detector through windows -> create heatmap -> treshold and segment heatmap -> bounding boxes

First, here is an image of the detector

![alt text][image5]

Now, the heatmap from this detection

![alt text][image6]

Finally, this is what the segmented image looks like

![alt text][image7]

---

###  Video Implementation

### #1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a video of the pipeline running

[![alt text][image8]](https://youtu.be/BoBsp6tZ4Pg)

### #2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded a heatmap class that increased the heat of the windows marked by the detector, but also implemented a decay term, that would reduce the heat of 
the entire image every iteration.

After this, I use label to create a bounding box along all contiguous parts of the image above a certain threshold.

###  Here is an example of the pipeline

![alt text][image5]

###  Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

###  Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

### #1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As I already mentioned in the Advanced Lane Finding project, this project shows the deficiencies of traditional computer vision vs end to end machine learning approaches.
There are a lot of parameters that you can tune, and small functions that improve the result. But, without an objective loss function, it is difficult to know if you are moving in the right direction.
Also, a computer would be much more efficient in tuning all these parameters in a neural network since 1.it's faster 2. it knows the derivative of all parameterers
and can move only in the right direction.
On top of it, running this program on the CPU is slow, it can only use 25% of even the CPU since is not parallelized, an parallelizing it would likely take weeks.
In contrast, and end to end deep learning approach runs on a massively parallel GPU

Apart from this, the result are good enough to detect cars quite consistently. I think there is room for improvement on creating smartes search boxes and in a smarter filter for the classifier.

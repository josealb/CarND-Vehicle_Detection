## Writeup


---

**Vehicle Detection Project**

In this file I will explain the steps I took to solve the problem of detecting cars on a video stream. The first version of the project used HOGs and SVMs and was rejected by the reviewer for missing frames where the cars were visible.
In this version, I switch to a convolutional neural network detection of the search window, which significantly improves accuracy, while reducing processing time.

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

First, the features are computed for the entire test and train dataset. Then, they are computed frame by frame for the video.

Before computing hog, I change the colorspace to LUV, which, as we saw in the latest project is useful for disregarding changes in luminance.
Even though hog is inherently more robust to different luminances, there is still value in having similar absolute values in classification.
To choose LUV I tried different colorspaces looking for the highest validation accuracy

![alt text][image2]

### #2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided it was best to use all channels for the hog, even though with hsv you could also just use the V channel, using more channels discarded some false positives.
I kept the standard parameters for orientation, number of pixels per cell, etc.

### #3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the following features:
* Histogram of gradients
* Raw pixel data (low resolution)
* Histogram of colors

The features are then stacked and scaled and fed into the classifier for training. This is done in line 65 (svc.fit(X_train, y_train))

### #3. As explained before, the method was changed to a convolutional neural network to improve the accuracy of the detection.

HOGs and SVM didn't show enough accuracy and tuning the parameters proved very difficult and time consuming. Since they run only on cpu every iteration of parameter change and test takes a while.

For this reason I introduced a neural network. In the first step I only substituted the SVM for a dense neural net, and kept extracing the same HOG and SVM features. This approach increased accuracy slightly, but not enough.

On the second step I decided to let the network learn the best features and used a convolutional neural network. This approach significantly improved the accuracy of window detection to 99.7% up from 97% of the SVM

The structure of the network is the following

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_41 (Convolution2D) (None, 64, 64, 64)    1792        convolution2d_input_2[0][0]      
____________________________________________________________________________________________________
convolution2d_42 (Convolution2D) (None, 64, 64, 64)    36928       convolution2d_41[0][0]           
____________________________________________________________________________________________________
activation_50 (Activation)       (None, 64, 64, 64)    0           convolution2d_42[0][0]           
____________________________________________________________________________________________________
maxpooling2d_33 (MaxPooling2D)   (None, 32, 32, 64)    0           activation_50[0][0]              
____________________________________________________________________________________________________
convolution2d_43 (Convolution2D) (None, 32, 32, 128)   73856       maxpooling2d_33[0][0]            
____________________________________________________________________________________________________
activation_51 (Activation)       (None, 32, 32, 128)   0           convolution2d_43[0][0]           
____________________________________________________________________________________________________
maxpooling2d_34 (MaxPooling2D)   (None, 16, 16, 128)   0           activation_51[0][0]              
____________________________________________________________________________________________________
convolution2d_44 (Convolution2D) (None, 16, 16, 256)   295168      maxpooling2d_34[0][0]            
____________________________________________________________________________________________________
activation_52 (Activation)       (None, 16, 16, 256)   0           convolution2d_44[0][0]           
____________________________________________________________________________________________________
maxpooling2d_35 (MaxPooling2D)   (None, 8, 8, 256)     0           activation_52[0][0]              
____________________________________________________________________________________________________
convolution2d_45 (Convolution2D) (None, 8, 8, 512)     1180160     maxpooling2d_35[0][0]            
____________________________________________________________________________________________________
activation_53 (Activation)       (None, 8, 8, 512)     0           convolution2d_45[0][0]           
____________________________________________________________________________________________________
maxpooling2d_36 (MaxPooling2D)   (None, 4, 4, 512)     0           activation_53[0][0]              
____________________________________________________________________________________________________
flatten_10 (Flatten)             (None, 8192)          0           maxpooling2d_36[0][0]            
____________________________________________________________________________________________________
dense_35 (Dense)                 (None, 512)           4194816     flatten_10[0][0]                 
____________________________________________________________________________________________________
activation_54 (Activation)       (None, 512)           0           dense_35[0][0]                   
____________________________________________________________________________________________________
dense_36 (Dense)                 (None, 256)           131328      activation_54[0][0]              
____________________________________________________________________________________________________
activation_55 (Activation)       (None, 256)           0           dense_36[0][0]                   
____________________________________________________________________________________________________
dense_37 (Dense)                 (None, 128)           32896       activation_55[0][0]              
____________________________________________________________________________________________________
activation_56 (Activation)       (None, 128)           0           dense_37[0][0]                   
____________________________________________________________________________________________________
dense_38 (Dense)                 (None, 1)             129         activation_56[0][0]              
====================================================================================================
Total params: 5,947,073
Trainable params: 5,947,073
Non-trainable params: 0


### Sliding Window Search

### #1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search only the windows that contain the road, from pixel 400 downwards. This reduces execution time and the probability of false positives

### #2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline is as follows.

Create windows -> Run detector through windows -> create heatmap -> treshold and segment heatmap -> bounding boxes

First, here is an image of the detector

![alt text][image5]

Now, the heatmap from this detection, with threshold >= 1 

![alt text][image6]

Finally, this is what the segmented image looks like

![alt text][image7]

---

###  Video Implementation

### #1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a video of the pipeline running

[![alt text][image8]](https://www.youtube.com/watch?v=ttnjMEQAHyg)

### #2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded a heatmap class that increased the heat of the windows marked by the detector, but also implemented a decay term, that would reduce the heat of 
the entire image every iteration.

After this, I use label to create a bounding box along all contiguous parts of the image above a certain threshold.

###  Here is an example of the pipeline

![alt text][image5]

###  Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
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

For this reason, I ended up switching to convnets and getting significantly better results.

A possible improvement for the project would be discarding the sliding window approach and substitute it for and end-to-end deep learning approach.

Thank you for reading

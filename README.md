# **Self-Driving Car Engineer Nanodegree** 

## Luis Miguel Zapata

---

## Behavioral Cloning Project

This projects aims to develop a Deep Neural Network to clone car driving behaviour in a simulated environment.  

[image1]: ./screenshots/simulator.png "Simulator"
[image2]: ./screenshots/left.jpg "Left"
[image3]: ./screenshots/center.jpg "Center"
[image4]: ./screenshots/right.jpg "Right"
[image5]: ./screenshots/backwards.jpg "Backwards"
[image6]: ./screenshots/recovery.jpg "Recovery"



### 1. Data collection.

Udacity's team has developed a virtual environment based on Unity's Engine to simulate a self driving that can be controlled using python. Such simulator has two different ways of being controlled:

* Training mode: Allows an user to control the car and gather information at all times.
* Autonomous mode: Creates a gateway for the car to be controlled using python scripts.

![alt text][image1]

In training mode, images sequences from 3 simulated cameras as well as the steering commands input by the user are stored. This simulator has two different roads to drive on, but we are going to focus on the single lane road. 

Left image                 |  Center image             |  Right image  
:-------------------------:|:-------------------------:|:-------------------------: 
![][image2]                |  ![][image3]              |  ![][image4]

#### Datasets
In order to obtain a good driving behaviour different data collections are done. The first dataset corresponds to one lap to the single lane road, this road mostly has turns to the left which could create a biased behaviour when training and therefore it is necessary to create another dataset, in this case, in the same road but driving backwards. Finally in order to be able to recover and go back to the center of the lane when the car goes to the side of the road dataset is created, but only recording when the car is already in one of the sides of the lane and goes back to the center.

Driving forward            |  Driving backwards        |  Recovery
:-------------------------:|:-------------------------:|:-------------------------: 
![][image3]                |  ![][image3]              |  ![][image4]

### 2. Model Architecture and Training Strategy.

Using the found calibration parameters every incoming image from the camera is corrected using the following function.

```
undist = cv2.undistort(img, mtx, dist, None, mtx) # Undistore the image
```
This procedure will ensure that the calculations performed will correspond to real world measurements.


Original image             |  Corrected image 
:-------------------------:|:-------------------------:
![][image2]                |  ![][image3]

### 3. Results.
Next step is to obtain the edges of the image. For this the approached that suited the best was two combine the S channel thresholding from HLS color space along with the magnitud of the gradient of RGB images. 

### 4. Potential shortcomings

This algorithm relies highly in a good segmentation of the lane lines and even though the Saturation channel from HLS color space is more robust to light changing conditions, it is not certain that good lines are going to be obtained and that will not be affected by shades or other facts.

### 5. Possible improvements

In my opinion a better segmentation of the lane lines has to be done and possibly Convolutional Neural Networks for this task could be used. 

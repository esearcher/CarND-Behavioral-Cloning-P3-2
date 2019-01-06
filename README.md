# **Self-Driving Car Engineer Nanodegree** 

## Luis Miguel Zapata

---

**Behavioral Cloning Project**

This projects aims to develop a Deep Neural Network to clone car driving behaviour in a simulated environment.  

[image1]: ./screenshots/simulator.png "Simulator"
[image2]: ./screenshots/left.jpg "Left"
[image3]: ./screenshots/center.jpg "Center"
[image4]: ./screenshots/right.jpg "Right"
[image5]: ./screenshots/backwards.jpg "Backwards"
[image6]: ./screenshots/recovery.jpg "Recovery"
[image7]: ./screenshots/gray.jpg "Gray"
[image8]: ./screenshots/equalized.jpg "Equalized"
[image9]: ./screenshots/normalized.png "Normalized"
[image10]: ./screenshots/test_1.png "Test 1"
[image11]: ./screenshots/test_2.png "Test 2"
[image12]: ./screenshots/test_3.png "Test 3"
[image13]: ./screenshots/test_4.png "Test 4"
[image14]: ./screenshots/test_5.png "Test 5"


### 1. Data collection.

The dataset consists of 3 different pickle files containing images and labels for the training, the validation and the testing.

![alt text][image1]

Using these found parameters any image can be corrected as the images below.

Left image                 |  Center image             |  Right image  
:-------------------------:|:-------------------------:|:-------------------------: 
![][image2]                |  ![][image3]              |  ![][image4]

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

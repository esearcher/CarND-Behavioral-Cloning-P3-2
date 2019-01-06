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
In order to obtain a good driving behaviour different data collections are done. The first dataset corresponds to one lap to the single lane road, this road mostly has turns to the left which could create a biased behaviour when training and therefore it is necessary to create another dataset, in this case, in the same road but driving backwards. Finally in order to be able to recover and go back to the center of the lane when the car goes to the side of the road one last dataset is created, This one corresponds to the same road, but only recording when the car is already in one of the sides of the lane and goes back to the center and avoiding to record the moment when the car is going away from the center of the lane to any of the sides.

Driving forward            |  Driving backwards        |  Recovery
:-------------------------:|:-------------------------:|:-------------------------: 
![][image3]                |  ![][image5]              |  ![][image6]

### 2. Model Architecture and Training Strategy.

Using Keras a sequential model is created and the model defined is based in the Nvidia pilot net model. The initial images supplied by the cameras are 320 x 120 pixels, these images are fed into a convolutional neural network  with 5 convolutional layers, 3 dense layers and 2 preprocessing steps.

* Cropping 2D: Image is cropped by 70 pixels and 25 pixels from the top and bottom correspondingly.
* Normalization: The resulting image is normalized between -1 and 1 using a lamda layer.
* 2D Convolution: 3 Filters 5x5 with a stride of 2 by 2 and a ReLu activation. 
* 2D Convolution: 24 Filters 5x5 with a stride of 2 by 2 and a ReLu activation.
* 2D Convolution: 36 Filters 5x5 with a stride of 2 by 2 and a ReLu activation.
* 2D Convolution: 48 Filters 3x3 with a stride of 1 by 1 and a ReLu activation.
* 2D Convolution: 64 Filters 3x3 with a stride of 1 by 1 and a ReLu activation.
* Flatten: The convolution filters are set to a flatten layer.
* Multilayer perceptron: Output = 100.
* Multilayer perceptron: Output = 50.
* Multilayer perceptron: Output = 10.
* Multilayer perceptron: Output = 1. (Steering angle)

Notice that the output layer is a single number that corresponds to the steering angle and therefore no softmax layer is required making this a regression problem.

For training, the data gathered from the simulator is loaded, but as input not only the center images will be used, but also the left and right images in order to augment the data. The steering angle for the center image will be the one recorded by the simulator, but for the left and right image the angle will be the same plus and minus a correction factor determined experimentaly.

```
# Left images with angle correction
left = line[1]
images.append(cv2.imread(left)[...,::-1])
measurements.append(float(line[3]) + correction)
# Right images with angle correction
right = line[2]
images.append(cv2.imread(right)[...,::-1])
measurements.append(float(line[3]) - correction)
```

The loss function used will be the Mean of Squared Error between the predicted and resulting angle and the optmizer will be Adam. The data is splitted 80% for training and 20% for validation, being shuffled every time during 10 epochs.

```
model.compile(loss='mse', optimizer='adam')
model.fit(X,y, validation_split=0.2, shuffle=True, nb_epoch=10)
```
The resulting loss was 0.0133 for the training set and 0.0683 for validation meaning the training converged succesfully. On the other hand these metrics constantly decreased during the 10 epochs meaning that apparently there was no over fitting during training.

### 3. Results.
Next step is to obtain the edges of the image. For this the approached that suited the best was two combine the S channel thresholding from HLS color space along with the magnitud of the gradient of RGB images. 

### 4. Potential shortcomings

This algorithm relies highly in a good segmentation of the lane lines and even though the Saturation channel from HLS color space is more robust to light changing conditions, it is not certain that good lines are going to be obtained and that will not be affected by shades or other facts.

### 5. Possible improvements

In my opinion a better segmentation of the lane lines has to be done and possibly Convolutional Neural Networks for this task could be used. 

# Behavioral Cloning

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center-lane-driving]: ./examples/center-lane-driving.jpg "Center lane driving"
[recovery-1]: ./examples/recovery-1.jpg "Recovery 1"
[recovery-2]: ./examples/recovery-2.jpg "Recovery 2"
[recovery-3]: ./examples/recovery-3.jpg "Recovery 3"
[dirt-recovery-1]: ./examples/dirt-recovery-1.jpg "Dirt recovery 1"
[dirt-recovery-2]: ./examples/dirt-recovery-2.jpg "Dirt recovery 2"
[dirt-recovery-3]: ./examples/dirt-recovery-3.jpg "Dirt recovery 3"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a Convolution Neural Network with three convolutional layers with 5x5 kernel size (depths 24, 36, 48) and two convolutional layers with 3x3 kernel sizes (depths: 64) (model.py lines 95-106).

These convolutional layers are then flattened and the output is given to a suite of fully connected layers (1164, 100, 50, 100, 1) (model.py line 112-124).

The model has a cropping layer to remove some of the top and the bottom of the image (model.py line 88)

The data is then normalized and mean-centered in the model using a Keras lambda layer (model.py line 90).

We also resize the image 66x200, the size expected by the NVIDIA network (model.py line 92).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

I trained the network for 3 epochs. With more epochs the validation loss was increasing quite dramatically.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:

* center lane driving.
* recovering from the left and right sides of the road.
* counter-clockwise driving.
* driving on track 2.

For details about how I created the training data, see the section below.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one I used in the traffic sign recognizer project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model had a low mean squared error on both the training and validation set, implying that the model wasn't overfitting, but in the simulator the car was still going off-track.

I decided to generate more training data, the steps I took are outlined in section 3 below.

Even with this new training data the car was still going off-road so I decided to try a new model. In the Behavioral Cloning project videos a NVIDIA architecture was mentioned so I decided to try it.

I implemented the NVIDIA architecture as is, with only a small difference initially: I wasn't resizing the images to the size suggested in the paper. Even without this the model performed much better than the previous iteration, but the car was still going off track at some point.

Instead of trying to tune my hyper-parameters I tried generating more training data but this didn't help.

That's when I noticed a couple of things:
* I had somehow missed one of the fully connected layer (1164)
* I was using a linear activation functin in my fully connected layers. I changed it to a ReLU activation function, the same I was using in my convolutional layers.
* I wasn't resizing the images to the size specified in the NVIDIA paper. I added a lambda layer making use of the `tf.image.resize_images` function of Tensorflow.

I had also viewed a video from Ian Goodfellow about batch normalization (https://www.youtube.com/watch?time_continue=218&v=Xogn6veSyxA) and decided to add one layer of it after each of my convolutional and fully connected layers. It slightly improved the accruracy and made the network learn faster. I need to read more about it because I haven't fully understood how it works.

After fixing all these issues, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-124) consisted of a convolution neural network with the following layers:

* Cropping of the image (60px at the top and 25px at the bottom).
* Lambda layer to normalise and mean-center the image.
* Lambda layer to resize the image to 66x200.
* Convolutional layer with a depth of 24, a kernel size of 5x5, 2x2 strides and ReLU activation.
* Convolutional layer with a depth of 36, a kernel size of 5x5, 2x2 strides and ReLU activation.
* Convolutional layer with a depth of 48, a kernel size of 5x5, 2x2 strides and ReLU activation.
* Convolutional layer with a depth of 64, a kernel size of 3x3, 1x1 strides and ReLU activation.
* Convolutional layer with a depth of 64, a kernel size of 3x3, 1x1 strides and ReLU activation.
* Flattening of the output of the convolutional layers.
* Dense layer of size 1164, ReLU activation function.
* Dense layer of size 100, ReLU activation function.
* Dense layer of size 50, ReLU activation function.
* Dense layer of size 10, ReLU activation function.
* Final dense layer of size 1.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center-lane-driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the middle when it's off on the side of the roads. These images show what a recovery looks like starting from the left side :

![alt text][recovery-1]
![alt text][recovery-2]
![alt text][recovery-3]

Then I recorded about two laps in counter-clockwise driving in order to get more data points.

The model was performing fine with all this training data but I was still running into issues on part of the road with dirt on the sides. I recorded more training data with the vehicle recovering when there is dirt on the side:

![alt text][dirt-recovery-1]
![alt text][dirt-recovery-2]
![alt text][dirt-recovery-3]

I decided to augment it by generating a flipped version of each image and taking the opposite of the steering measurement.

Finally, I generated more training data by recording one lap on the second track.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.

Note: I tried to use a generator but consistently ran into an issue were the network was incredibly slow to train for the first epoch (couple of hours) so I abandoned the idea. I plan to come back to it to understand why this was the case, the hardware I use will not be able to hold my training data in memory indefinitely.


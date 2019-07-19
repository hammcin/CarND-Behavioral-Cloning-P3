# **Behavioral Cloning**

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_drive.jpg "Center Lane Image"
[image2]: ./images/recover1.jpg "Recovery Image Start"
[image3]: ./images/recover2.jpg "Recovery Image Middle"
[image4]: ./images/recover3.jpg "Recovery Image End"
[image5]: ./images/norm.jpg "Normal Image"
[image6]: ./images/flipped.jpg "Flipped Image"
[image7]: ./images/model_mse_loss.jpg "MSE Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 video of trained convolution neural network driving car autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture I used (model.py lines 62-108) was Nvidia's architecture for the same problem described [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

The architecture included 5 convolutional layers (model.py lines 70-88).  The first 3 convolutional layers had a kernel size of 5x5 with stride=(2,2) (model.py lines 70-80).  The first convolutional layer had 24 feature maps (model.py line 70), the second had 36 feature maps (model.py line 74), and the third had 48 feature maps (model.py line 78).  The final two convolutional layers had 3x3 kernels with stride=(1,1) (model.py lines 82-88).  The final two convolutional layers had 64 feature maps each (model.py lines 82-88).  Each convolutional layer was followed by a RELU activation function (model.py lines 70-88).

After the final convolutional layer, the features were flattened and fed to three successive fully connected-layers with 100, 50, and 10 units respectively before producing a single output (model.py lines 90-104).  Each fully-connected layer was also followed by a RELU activation function (model.py lines 92-102).

Finally, the first layer in the newtwork was a normalization layer (implemented as a lambda layer) which scaled the input by 255.0 and shifted the input input toward minus infinity by 0.5 (model.py line 66).  This was followed by a layer which cropped the top 70 rows of pixels and the bottom 25 rows of pixels from the image (model.py line 68).

As a modification to the NVIDIA architecture, I inserted Batch Normalization layers after each convolutional layer but before the RELU activation function.  I also added Batch Normalization layers after each fully-connected layer, also before the RELU activation (model.py lines 70-102).  This choice was inspired by the discussion [here](https://medium.com/@erikshestopal/udacity-behavioral-cloning-using-keras-ff55055a64c).

#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting (model.py lines 70-102).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 139-144). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 63 & 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving around the track in both the clockwise and counterclockwise directions.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the Nvidia convolution neural network model for predicting steering angles.  I thought this model might be appropriate because of the success Nvidia had with this model in predicting steering angles from images which was exactly the problem I was trying to solve.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding batch normalization layers.  Batch normalization is known to have a regularizing effect and to remove the need for dropout layers completely in some cases.  An additional benefit is that batch normalization speeds up training.

Then I retrained the model and found that I had a low mse on both the training and validation sets.  

The final step was to run the simulator to see how well the car was driving around track one. I noticed that the vehicle was falling off the track on curves, especially on parts of the track surrounded by brown dirt.  To improve the driving behavior in these cases, I collected more data.  I recorded another lap around track 1 where I focused on center lane driving around the curves in the track.  I also recorded two additional laps (clockwise and counterclockwise directions) of recovery driving where I tried to make sure I recorded recovery driving away from brown dirt areas toward the center of the track.  I also tried to make sure I had pleny of recovery driving on curves for my training data.

However, after gathering this additional training data, I still noticed the car seemed to be driving off the track on the steepest curve.  This motivated me to record recovery driving data of the car recovering from getting too close to this curve specifically.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes (model.py lines 62-108):

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 160x320x3 RGB image  						              |
| Normalization         | x/255 - 0.5                                   |
| Cropping              | Crop to size 65x320x3                         |
| Convolution 5x5     	| 2x2 stride, 24 feature maps               	  |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Convolution 5x5     	| 2x2 stride, 36 feature maps               	  |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Convolution 5x5     	| 2x2 stride, 48 feature maps               	  |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Convolution 3x3     	| 1x1 stride, 64 feature maps               	  |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Convolution 3x3     	| 1x1 stride, 64 feature maps               	  |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Flatten               |                                               |
| Fully connected		    | 100 units                        				      |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Fully connected		    | 50 units                        				      |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Fully connected		    | 10 units                        				      |
| Batch Normalization   |                                               |
| RELU					        |												                        |
| Output				        | 1 unit								                        |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one in the clockwise direction and one lap on track one in the counterclockwise direction using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its course if it drifted off to the side of the road. These images show what a recovery looks like starting from the left side of the road:

![alt text][image2]
![alt text][image3]
![alt text][image4]

In total I recorded two laps of recovery driving around track one in the counterclockwise direction and one lap of recovery driving around track one in the clockwise direction.

To augment the data sat, I also flipped images and angles thinking that this would balance the dataset so that the model learns to make both left and right turns. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had 54,750 data points. I then preprocessed this data by dividing each pixel value by 255.0 and subtracting 0.5 from each pixel value.  I also cropped the top 70 rows and bottom 25 rows of pixels from each image.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by the mean squared error of 0.0018 on the validation set:

![alt text][image7]

I used an adam optimizer so that manually tuning the learning rate wasn't necessary.

### Simulation

#### 1. Car Navigation on Test Data

Here's a [link to my video result](./video.mp4).

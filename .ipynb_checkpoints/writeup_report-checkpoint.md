# **Behavioral Cloning** 
---

## **Behavioral Cloning for Autonomous Driving using ConvNets in Keras**

**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/anticlk_good_center_2019_01_28_06_59_43_127.jpg "Anticlockwise Good Ride"
[image2]: ./examples/clk_center_2016_12_01_13_33_02_255.jpg "Clockwise Good Ride"
[image3]: ./examples/recovery_images_sample.png "Left Laneline Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the jupyter notebook script to create and train the model
* utils.py with utility functions for reading, writing , and plotting images
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing a video recording of the vehicle driving autonomously over one lap around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.ipynb jupyter notebook contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I had 27686 training images and used transfer learning to build a model using the pre-trained VGG16 network available in Keras. The model code is defined in the function `get_vgg_based_model` in the model.ipynb, section `VGG Based Model`. 

My convolution neural network model consists of the following:

* Convolution layers with 3x3 filter sizes and depths between 64 and 512
* Max-Pooling layers with 2x2 filters
* Zero-Padding layers (1x1)
* Dropout Layers
* Batch Normalization Layers
* Fully-connected (Dense) layers with depths between 4096 and 1

The model includes RELU layers to introduce nonlinearity. The input image data, which is 160x320x3, are preprocessed by subtracting the RGB channel means calculated on the imagenet data by the VGG16 model creators, and then reversing the RGB channels to BGR as expected by the pre-trained VGG16 network. This is done using a Lambda layer. Further, before feeding the image data to the first convolution layer, the input image is cropped by 50 and 20 lines from top and bottom respectively, to produce the final imput image with dimensions 90x320x3. This is done to remove the landscape and car-hood portion and keep only the track/road portion in the image.

More details on the model architecture to follow.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, and was trained and validated on different data sets to ensure that the model was not overfitting. The model training code and output can be found in `model.ipynb` section `Model Training`. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The video of the final simulator test run can be found in the `video/` folder.

#### 3. Model parameter tuning

The model used an adam optimizer. Although adam oprimizer uses adaptive learning rate for individual parameters, instead of relying on it, I jumped the learning rate lower, manually, after 10, 20 , and 40 epochs and the results looked good. The code can be found in `model.ipynb` section `Model Training`.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road surface, between the track/lane lines. I used a combination of images with center lane driving, recovering from the left and right sides of the road, and with the vehicle driving in both, clockwise and anti-clockwise, directions.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I thought of starting with using only the center camera images first and adding the left and right camera images later, if required, but then center camera images were sufficient to produce the desired driving performance. Hence, I did not have to use the side camera images. I had a total of 34608 center camera images and my overall strategy for deriving a model architecture was to take advantage of transfer learning, using the simple and elegant VGG16 model architecture.

I used the pretrained VGG16 model in Keras without including the top (Dense layers). The VGG16 model, which has 5 convolution blocks with 2, 2, 3, 3, and 3 colvolution layers, respectively, in them, is trained to recognize images from 1000 different categories. The earlier network layers, which find more simple features such as edges, simple shapes, color, etc. would be more relevant as starting point for this application, than the later layers which find more complex shapes such as objects, human/animal faces, etc. So, I experimented with freezing weights for the first 2 and 3 convolution blocks and making the weights for the later blocks trainable. I found that freezing the first 2 blocks, rather than 3 blocks, gave me better results. After the convolution layers, I added a Flattening layer and 4 Fully-connected layers with 4096, 4096, 2048, and 1024 units, followed by a Dense layer with 1 unit for the steering angle prediction. All the Convolution and Dense layers had the RELU activation for introducing nonlinearity, except for the final 1 unit Dense layer which outputed the steering angle as a regression output. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation sets with 27686 (80%) and 6922 (20%) images respectively. My initial goal was to, without using any regularization technique, overfit the model so as to ensure that I have a good enough model architecture. My initial training runs of the model had a low mean squared error on the training set but a high mean squared error on the validation set, which indicated that the model was overfitting. Then, I added Dropout layers, with 50% dropout rate, as a regularization technique, which gave me good results. I also added BatchNormalization layer to the Dense layers.

After every few epochs (multiples of 10), I tested the model by running it on the simulator. Initially, I was using only the lane-center driving behavior images to train the initial model, which took the car off the track a few times. So, I generated and added a set of images that exhibited recovery behavior and that took care of the car going really off the track. Though, the vehicle did go a little offtrack before recovering. Training  the model for more epochs took care of that.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. The video of the simulator test run can be found in the `video/` folder.

#### 2. Final Model Architecture

My final model, which is defined in the function `get_vgg_based_model` in the model.ipynb, section `VGG Based Model`, has the following architecture:

| Layer (type)   				| Output Shape 		| # Params 	|
|:---------------------------------------------:|:---------------------:|:-------------:| 
| image_input (InputLayer) 			| (None, 160, 320, 3) 	|	0	|
| lambda_1 (Lambda) 				| (None, 160, 320, 3) 	|       0       |
| cropping2d_1 (Cropping2D) 			| (None, 90, 320, 3) 	|       0	|
| block1_conv1 (Conv2D) 			| (None, 90, 320, 64) 	|       1792    |
| block1_conv2 (Conv2D) 			| (None, 90, 320, 64) 	|       36928   |
| block1_pool (MaxPooling2D) 			| (None, 45, 160, 64) 	|       0       |
| block2_conv1 (Conv2D) 			| (None, 45, 160, 128) 	|      	73856   |
| block2_conv2 (Conv2D) 			| (None, 45, 160, 128) 	|      	147584  |
| block2_pool (MaxPooling2D) 			| (None, 22, 80, 128) 	|       0       |
| block3_conv1 (Conv2D) 			| (None, 22, 80, 256) 	|       295168  |
| block3_conv2 (Conv2D) 			| (None, 22, 80, 256) 	|       590080  |
| block3_conv3 (Conv2D) 			| (None, 22, 80, 256) 	|       590080  |
| block3_pool (MaxPooling2D) 			| (None, 11, 40, 256) 	|       0       |
| block4_conv1 (Conv2D) 			| (None, 11, 40, 512) 	|       1180160 |
| block4_conv2 (Conv2D) 			| (None, 11, 40, 512) 	|       2359808 |
| block4_conv3 (Conv2D) 			| (None, 11, 40, 512) 	|       2359808 |
| block4_pool (MaxPooling2D) 			| (None, 5, 20, 512) 	|       0      	|
| block5_conv1 (Conv2D) 			| (None, 5, 20, 512) 	|       2359808	|
| block5_conv2 (Conv2D) 			| (None, 5, 20, 512) 	|       2359808	|
| block5_conv3 (Conv2D) 			| (None, 5, 20, 512) 	|       2359808	|
| block5_pool (MaxPooling2D) 			| (None, 2, 10, 512) 	|       0      	|
| dropout_1 (Dropout) 				| (None, 2, 10, 512) 	|       0      	|
| flatten_1 (Flatten) 				| (None, 10240) 	|       0 	|
| fc1 (Dense) 					| (None, 4096) 		|	41947136|
| batch_normalization_1 (Batch Normalization) 	| (None, 4096) 		|	16384	|
| dropout_2 (Dropout) 				| (None, 4096) 		|	0	|
| fc2 (Dense) 					| (None, 4096) 		|	16781312|
| batch_normalization_2 (Batch Normalization) 	| (None, 4096) 		|       16384   |
| dropout_3 (Dropout) 				| (None, 4096) 		|       0       |
| fc3 (Dense) 					| (None, 2048) 		|       8390656 |
| batch_normalization_3 (Batch Normalization) 	| (None, 2048) 		|       8192    |
| dropout_4 (Dropout) 				| (None, 2048) 		|       0       |
| fc4 (Dense) 					| (None, 1024) 		|       2098176 |
| batch_normalization_4 (Batch Normalization) 	| (None, 1024) 		|       4096    |
| dropout_5 (Dropout) 				| (None, 1024) 		|       0       |
| y_hat (Dense) 				| (None, 1) 		|       1025    |

**Total params:** 83,978,049

**Trainable params:** 83,695,361

**Non-trainable params:** 282,688

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded over four laps in anticlockwise direction on track one using center lane driving. Udacity had already provided sample image data that show good driving behavior in the clockwise direction. I added that to the good driving behavior data in order to prevent the model from having an anticlockwise direction bias.

Another way to avoid anticlockwise direction bias, would have been to augment the data with left-right flipped images. I tried using this approach, but it wasn't of much help, probably because driving data for both, clockwise and anticlockwise, directions was already present. A custom image data generator that I implemented and used to augment the data can be found in the function `trainGeneratorFunction` in model.ipynb, section `Data Generation and Augmentation`. 

Here is an example image of center lane driving in both, clock wise and anitclock wise directions:

Anti-clockwise Good Riding | Clockwise Good Riding
:-------------------------:|:----------------------------:
![][image1]                |![][image2]

I then recorded the vehicle recovering from the left and right sides of the road back to center, so that the model would learn to prevent the vehicle from going off the track and/or would recover in time in case it does drive over the track/lane lines. I also recorded other recovery cases, such as steering back to the center of the lane in case the vehicle is heading left or right while approaching or getting off the bridge.

Below images, titled with their steering angles, show what a recovery looks like starting from the vehicle on the left track and driving back to the center of the track:

![alt text][image3]

After the data collection process, I had a total of 34608 images. I then read the images, and their respective steering measurements, using the `driving_log.csv` generated by the simulator. I shuffled the data, and saved it into a `h5py` file for future loading. The implementation can be found in `model.ipynb`, in section `Prepare and Save Dataset`.

After loading the already shuffled data, I divided it into two sets of 27686 (80%) and 6922 (20%) images as training and validation data respectively, the implementation of which can be found in the `model.ipynb`, section `Load Dataset`. 

I used this training data for training the model. The validation set helped determine whether the model was over or under fitting. The model training took anywhere between 522 to 760 seconds for 1 epoch on my local machine with a Nvidia GTX1070 GPU. As we can see from the training output under section `Model Training` in `model.ipynb`, the ideal number of epochs was around 40. Although the model at 40 epochs shows a little bit of overfitting, it probably isn't much, as it showed the best simulation test results when driving the car in autonomous mode. I used this model at 40 epochs to generate the video and it is also the one renamed to `model.h5` for submission, as required by the project naming conventions.

As an observation, the `model_ep60.h5`, when run on the simulator, showed few jerky vehicle movements when approaching the sides, or turning, and also drove over the lane lines at a few locations. If we see in the training log, after 50 epochs (Epoch 10/20 in the ep41-60 run output), the overfitting seems to be increasing, and this could possibly be the reason for the jerky movements I saw in the autonomous drive test for model_ep60.h5.

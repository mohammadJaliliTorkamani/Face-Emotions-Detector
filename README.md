### Python Face Emotion Detetor (OpenCV + Cascade Classifier)
an open-source console application developed with `Python 3` using `OpenCV`, `Keras`, and `Cascade classifier` to train and detect seven human face emotion types as follows below:
* Angry
* Happy
* Disgust
* Sad
* Scared
* Surprised
* Neutral

<br/>

## Requirements
- Python (version 3) and pip: follow [this link](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-ubuntu-20-04-server) to install python3 and pip library on your computer.
- OpenCV:

```bash
sudo pip3 install opencv-python
```

- tensorflow (GPU version):

```bash
sudo pip3 install tensorflow-gpu
```
- Keras

```bash
sudo pip3 install keras
```
- imutils

```bash
sudo pip3 install imutils
```
- NVidia, Cuda and cuDNN: If you have not already installed your graphic card drivers, you can use [this link](https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d "this link") to install Nvidia drivers and configure your Cuda and cuDNN.

<br/>

## Training face detector
First of all, extract **Datasets.zip** into the main project directory, then follow thes steps:

###step 1 - Collecting Dataset
First of all, You need to use the cascade classifier method to detect human faces. You can find a variety of datasets on the internet for face detection. Look at [here](http://vision.ucsd.edu/content/yale-face-database) or [here](http://vision.ucsd.edu/content/extended-yale-face-database-b-b) for instance and download a suitable dataset. You can use the cropped faces for positive examples. (the cropped faces are either available in the datasets, or you need to extract them using bounding boxes and resize them for training).

###step 2 - Creating training data files
You need to create postives.txt and negatives.txt files using the commands below:
```bash
find ./negative_images -iname "*.pgm" > negatives.txt
find ./positive_images -iname "*.jpg" > positives.txt
```

###step 3 - Craeting sample
First, use the createsamples.pl file (located in the Final directory) to create a `.vec` file for each dataset image. The output of this command is a set of `.vec` files, a binary format that contains images:
```bash
perl createsamples.pl positives.txt negatives.txt ../samples 5000 "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 40 -h 40"
```

Second, you need to merge all these `.vec` samples to create a single file:
```bash
python mergevec.py -v samples/ -o samples.vec
```
<br/>

###step 4 - Training Local Binary PAttern (LBP) cascade
As you know, LBP is much faster than Haar but is less accurate. We use this method to train our detector as below:
```bash
opencv_traincascade -data lbp -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 4000 -numNeg 7000 -w 40 -h 40 -mode ALL -precalcValBufSize 4096 -precalcIdxBufSize 4096 -featureType LBP
```
Soon after entering the command, the training operation to create a cascade detector starts, and after a while, you have a file named 'cascade.xml', which You can use to detect human faces.
<br/>

## Training Facial Expression Classification
Run `neural_network_classifier_default.py` to start training of the dataset:
```bash
python neural_network_classifier_default.py
```
After a while, you have some models file with `.hdf5` extension in your models directory located in Code directory. Stop training when you have the least loss and max accuracy. We have used  `_mini_XCEPTION.75-0.64.hdf5` weight as our main weight file, but you can use your desired file instead. Note that you have to change the `weight_file_address` variable in `completed.py` to use another weight file for detection. 

## Running
Open your terminal in the project directory and enter this command:
```bash
python3 ./Code/completed.py
```

<br/>

## Output:
![](https://github.com/mohammadJaliliTorkamani/Face-Emotions-Detector/blob/master/media/ezgif.com-gif-maker.gif)

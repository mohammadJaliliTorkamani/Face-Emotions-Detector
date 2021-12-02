### Python Face Emotion Detetor (OpenCV + Cascade Classifier)
an open-source console application developed with `Python 3` using `OpenCV`, `Keras`, and `Cascade Classifier` to train and detect seven human face emotion types as follows below:
* Angry
* Happy
* Disgust
* Sad
* Scared
* Surprised
* Neutral

<br/>

## Requirements
- Python3 and pip: Follow [this link](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-ubuntu-20-04-server) to install python3 and pip library on your computer.

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

### step 1 - Extracting the Dataset
Extract **Datasets.zip** into the main project directory

### step 2 - Creating training data files
You need to create postives.txt and negatives.txt files using the commands below:
```bash
find ./negative_images -iname "*.pgm" > negatives.txt
find ./positive_images -iname "*.jpg" > positives.txt
```

### step 3 - Craeting samples
First, use the createsamples.pl file (located in Final directory) to create a `.vec` file for each dataset image. The output of this command is a set of `.vec` files, a binary format that contains images:
```bash
perl createsamples.pl positives.txt negatives.txt ../samples 5000 "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 40 -h 40"
```

Then, you need to merge all these `.vec` samples to create a single file:
```bash
python mergevec.py -v samples/ -o samples.vec
```
<br/>

### step 4 - Training Local Binary Pattern (LBP) cascade
Although LBP algorithm is much faster than Haar, it is less accurate. We aim to have a fast but less-accurate application so we use this algorithm to train our detector:
```bash
opencv_traincascade -data lbp -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 4000 -numNeg 7000 -w 40 -h 40 -mode ALL -precalcValBufSize 4096 -precalcIdxBufSize 4096 -featureType LBP
```

<br/>
Soon after entering the command, the training operation starts to create a cascade detector. After a few hours, you will have a file named 'cascade.xml', which can be used to detect human faces.
<br/>

## Training Facial Expression Classification
Run **neural_network_classifier_default.py** to start training of the dataset:
```bash
python neural_network_classifier_default.py
```
After a while, you have some models file with `.hdf5` extension in your models directory located in Code directory. Stop training when you have the least loss and max accuracy. We have used  "_mini_XCEPTION.75-0.64.hdf5"  weight as our main weight file, but you can use your desired file instead. Note that you have to change the **weight_file_address** variable in completed.py in order to use another weight file.

## Running
Open your terminal in the project directory and enter this command:
```bash
python3 ./Code/completed.py
```

<br/>

## Output:
![](https://github.com/mohammadJaliliTorkamani/Face-Emotions-Detector/blob/master/media/ezgif.com-gif-maker.gif)


### Link Resources
| URL |
| ------------ |
| https://towardsdatascience.com |
| https://www.digitalocean.com |


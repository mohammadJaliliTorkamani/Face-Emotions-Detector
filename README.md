### Python Face Emotion Detetor (OpenCV + Cascade Classifier)
an open-source console application developed with `Python 3` using `OpenCV`, `Keras` and `Cascade classifier` to train and detect seven human face emotion types as follows below:
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

## Training
First of all, you need to  use cascade classifier method to detect human faces. we have  do that before and saved the final .xml file in order to use as your cascade source file. 

<br/>

## Running
Open your terminal in the project directory and enter this command:
```bash
python3 ./Code/completed.py
```

<br/>

## Output:
![](https://github.com/mohammadJaliliTorkamani/Face-Emotions-Detector/blob/master/media/ezgif.com-gif-maker.gif)

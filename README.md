# volumeByDepth

This script can be used to setup an oak-d or oak-d pro and use its depth images to calculate a resulting volume

## Installation

### linux (ubuntu):

#### install python and opencv:
apt update<br />
apt -y upgrade 

apt-get install -y python3.10<br />
apt-get install -y python3.10-distutils<br />
apt-get install -y python3-pip<br />
apt-get install -y python3-opencv

#### install other requirements
python3.10 -m pip install -r requirements.txt

### windows

#### download and install python
python3.10 from: https://www.python.org/downloads/release/python-31011/

#### install requirements
python3.10 -m pip install -r requirements.txt

#### install opencv
download opencv-4.8.0-windows.exe from https://github.com/opencv/opencv/releases and install it 

## Usage

### start the programm
python depth.py

### shortcuts
|   shortcut   |    function    |
|--------------|----------------|
| q| quit |
| w|  save 1 image| 
| r|  calc volume of the saved images| 
| t|  save 10 images (one every 2 seconds)|
| e|  save depth image in a nice way|
| z|  set base volume and delete processed images|
| u|  delete images|
| i|  calc difference to base volume|
| p|  print pixel value at some positions|
| f|  enter FloorMode:  double-click on the image to mark the floor area|
| v|  enter ValidMode:  double-click on the image to mark the valid area|
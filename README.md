# volumeByDepth

## Installation

### linux (ubuntu):

#### install python and opencv:
apt update
apt -y upgrade 

apt-get install -y python3.10
apt-get install -y python3.10-distutils
apt-get install -y python3-pip
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
| w|  safe 1 image| 
| r|  calc volume of the images| 
| t|  write 10 images|
| e|  print depth image in a nice way|
| z|  set base volume|
| u|  delete images|
| i|  calc difference to base volume|
| p|  print pixel value at some positions|
| f|  enter FloorMode:  double-click on the image to mark the floor area|
| v|  enter ValidMode:  double-click on the image to mark the valid area|
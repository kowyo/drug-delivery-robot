# Drug Delivery Robot

## Introduction

This is a robot that can be used to deliver drugs to patients in a hospital. The robot will drive through the sparse cones and follow the nurse in the end of the road.

The function implemented is not complicated. However, it is a good pratice of how to integrate YOLOv5 (or other deep learning models) into ROS and how to use Python to write ROS packages.

## Features
- Crafted entirely in Python.

- Integrate YOLOv5 into ROS and used it to detect the cones in the image published by the camera.

- Color mask implemented in OpenCV is used to detect the nurse.

## Installation

### Prerequisites

- Ubuntu 20.04

- ROS Noetic

- Python 3.8 (Comes with [ROS Noetic installation](http://wiki.ros.org/noetic/Installation/Ubuntu))

- zsh (It is recommended to use zsh as it has better auto-completion than bash)

It is recommended to have a basic understanding of ROS Topics. The following tutorial will walk you through the process of writing a simple publisher and subscriber.

- [Writing a Simple Publisher and Subscriber (Python)](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29)

### Dependencies

> [!NOTE]  
> A virtual environment is not recommended for ROS as it does not officially support this feature. All python packages should be installed globally using `pip install`. If you cannot find pip, you can install it by running:
> ```zsh
> sudo apt install python3-pip
> ```

#### YOLOv5

You can install YOLOv5 by running the following the instructions in the [YOLOv5 repository](https://github.com/ultralytics/yolov5)

If you are using a custom model, you need to modify the 'listener.py' file to load your model.

```python
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt') # local model
# or
model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo
```

#### OpenCV

To use OpenCV in ROS, you need to refer to [this tutorial](https://index.ros.org/p/cv_bridge/) to install OpenCV.

Note that 'opencv2' is not required to be included in the `find_package` command in `CMakeLists.txt` as it is already included in `cv_bridge`.

#### PyTorch and other python packages

They should be installed with ease using `pip install`.

## Usage

This is a ROS package. To use it, you need to clone this repository to your catkin workspace and run `catkin_make`.

For example, if your catkin workspace is `~/catkin_ws`, you can run the following commands to clone this repository and build it:

```zsh
cd ~/catkin_ws/src # Go to your catkin workspace
catkin_create_pkg your_package_name # Create a new package
cd your_package_name # Go to your new package
rm rf * # Remove all files in the package
git clone https://github.com/xiln7/drug_delivery_bot
cd ~/catkin_ws # Go back to your catkin workspace
catkin_make # Build the package
```

After building the package, you can run the following command to launch the robot:

To launch the camera and publish the image

```zsh
cd ~/catkin_ws
source devel/setup.zsh
rosrun your_package_name talker.py
```

To subscribe to the image and detect the cones

```zsh
cd ~/catkin_ws
source devel/setup.zsh
rosrun your_package_name listener.py
```

## Demo

https://github.com/xiln7/drug_delivery_bot/assets/110339237/c0b99b8b-1e8a-487d-aca7-a1e24230e809

## Citation

If you use this repository in your research, please cite our work as follows:

```bibtex
@software{Drug Delivery Robot,
  title = {Drug Delivery Root},
  author = {Zifeng Huang, Shunyu Zhou},
  year = {2023},
  url = {https://github.com/xiln7/drug_delivery_bot}
}
```



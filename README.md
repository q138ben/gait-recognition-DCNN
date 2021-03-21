# gait-recognition-DCNN


Gait phase recognition is of great importance in the development of assistance-as-needed
robotic devices, such as exoskeletons. In order for a powered exoskeleton with phase-based control
to determine and provide proper assistance to the wearer during gait, the user’s current gait phase
must first be identified accurately. Gait phase recognition can potentially be achieved through input
from wearable sensors. Deep convolutional neural networks (DCNN) is a machine learning approach
that is widely used in image recognition. User kinematics, measured from inertial measurement unit
(IMU) output, can be considered as an ‘image’ since it exhibits some local ‘spatial’ pattern when the
sensor data is arranged in sequence. We propose a specialized DCNN to distinguish five phases in
a gait cycle, based on IMU data and classified with foot switch information. The DCNN showed
approximately 97% accuracy during an offline evaluation of gait phase recognition. Accuracy was
highest in the swing phase and lowest in terminal stance.

## The purpose of this repository

The purpose of this repository was to demonstrate the labelling of the sensor data with corresponding gait phases, training with a CNN network, and cross validate the recognition accuracy.

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:42:41 2018

@author: binbi
"""


#
import socket
import struct
import numpy as np
from keras.models import load_model
from make_matrix import make_matrix_7by9

fs = 2000; # Myon Sample time 2000hz 

# Set for how long the live processing should last (in seconds)
endTime = 10; # Set the time of acquistion here
frameCount = 0;
channel = 63
#RawIMU = np.zeros((fs*endTime,channel)); 
RawIMU = np.zeros((1,channel)); 
EMGwindow = 0;
n = 480; #Buffer size
p = 20; # Buffer overlap
k = 0; #Used to save the RawEMG data

TCP_IP = '127.0.0.1'
TCP_PORT = 5000
BUFFER_SIZE = 1024


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


model = load_model('best_model_allData.h5')


i=0
while True:
    
    
    dataRecv = s.recv(252)
    data = struct.unpack('63f',dataRecv)
    RawIMU[:] = data # 
    X = make_matrix_7by9(RawIMU)
    X = X.reshape(X.shape[0],7,9,1)
    y_pred= model.predict_classes(X)
#        print (dataRecv)
#    print ('received data',data)
    print ('y_pred = ',y_pred)

 #       break
#s.close()
 











#fs = 2000; # Myon Sample time 2000hz 
#
## Set for how long the live processing should last (in seconds)
#endTime = 30; # Set the time of acquistion here
#frameCount = 1;
#RawEMG = np.zeros(1,fs*endTime); 
#EMGwindow = 0;
#
#while(frameCount/fs <= endTime):
#   RawEMG(frameCount) = read(t, 1, 'single'); % Read one sample at time
#   EMGwindow = EMGwindow+1; % Collect samples in buffer of size n
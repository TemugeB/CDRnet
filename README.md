# Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation

![Alt text](imgs/kpts_preds.png?raw=true "keypoint_predictions")
![Alt text](imgs/3d_prediction.png?raw=true "3d_predictions")

This is a Tensorflow implementation of the paper "Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation" by Remelli, et al[1]. 
The network is trained on the Human3.6M data set[2]. I'm not part of the authors and this implementation is an independent implementation.  

Here you will find the training code for the canonical fusion model described in the paper. This implementation is based on the paper only so some minor details are different from the author's intention. The code here will let you train end to end. I've also included testing code as well as a trained model. However, you will need to download the Human3.6M data. I wrote this implementation for some personal testing. But since the authors did not release an official implementation, I've uploaded mine for anyone interested in testing it. 

**Requirements**:  
-Tensorflow 2.4.1  
-cdflib  
-opencv  
-matplotlib  
-scipy  
The requirements can all be installed through pip.

**How-to**  
-Clone the repository.  
-Download Human3.6M data.  
-Place video data into Data folder and label data into Labels folder.  
-Download the trained model if you want to test.  
-Run the train_and_test.py script. Choose train or test in the script.  

**Training**  
The network was trained on a GTX1080Ti. It will take about 9 to 12 hours of training. You should be able to get the loss value down to 2.4~2.5 MSE. The differentiable DLT layer descirbed in the paper is included but not used to calculate a loss. I found serious instability issues in using the layer. But the code is included if you want to use it. Simply replace _dummy_loss with _DLT_loss in the model.compile() call. 

**Note:**  
Subject9 has some incorrect labels that will significantly hinder network training. Make sure to remove these wrong labeled data. Please check the S9_training_vids.txt for the list of videos that can be used for training.

**References**:  
1. Edoardo Remelli, Shangchen Han, Sina Honari, Pascal Fua, Robert Wang. Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation. 	arXiv:2004.02186

2. Catalin Ionescu, Dragos Papava, Vlad Olaru and Cristian Sminchisescu, Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 36, No. 7, July 2014



**11-785, Spring 2020, Homework 1 Part 2**

***Frame-level Classification of Speech***

Link: https://www.kaggle.com/c/11-785-s20-hw1p2/leaderboard

Team Name: clearlove7


Account Name: Zhetian Jin


2/318 on private board

========================================================

Note:

1. PCA decomposition


preprocess the features to 10 features or 40 features before into the ContextDataset


2. Parameter Scheduler


decaying the learning rate


3. 8-layer network


normal network (linear + batchnorm + relu), and also add avg pool and max pool for the last layer.


4. Cost about 30 minutes for one epoch, 10 epochs will reach to about 68% accuracy on validation dataset


The result is shown in the notebook.


5. Padding methods


I pad each utterance with the first frame and last frame. For example, if there are 422 frames in the same utterance.
0 1 2 3 4 5 ....420  421  =====> 0 0 0 0 0 0 0 0 1 2 3 4 5 ....420 421 421 421 421 421 421 421 421 

 
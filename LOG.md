# Project log

This is a log of any tasks completed throughout the project.

Keywords:
* CNN = Convolutional Neural Network
* RBC = Red Blood Cell
* SVM = Support Vector Machine
* MPL = Multi-Layer Perceptron

01/06/2017
----------

> Refactor DiagnosisSystem
> * All configs are now handled directly by DiagnosisSystem

> Classified ~ 2000 cells, 74 contained P. falciparum

> Combined config files into one

> Imported images into numpy array for training the CNN
> * Some trainin images are less than 80x80 due to being on the edge of the main image
> * Should they be ignored?
> * Or padded with zeros?

30/06/2017
----------

> Classified ~ 2000 cells, 77 contaiend P. falciparum

> Began writing code to import training images and organise for CNN

29/06/2017
----------

> Implementing CNN for cell classification. 
> * Write function to train the CNN.
> * Write function to evaluate the CNN.
> * Write function to test unclassified images.

> Wrote code to aid manual classification of cells.

28/06/2017
----------

> Followed Keras tutorial.
> Implemented a CNN deep learning model in Keras for CIFAR-10.

> Collected sample images from 57546.


27/06/2017
----------
> Write interim report.

> Submit interim report.

> Followed Keras tutorial.
> Implemented a MLP deep learning model in Keras for NMIST.

26/06/2017
----------
> Still writing Interim Report.

25/06/2017
----------
> Began writing Interim Report.

21/06/2017
----------
> Refactored code:
> * Move main to src directory.
> * Re-write _load_samples method.
> * Load samples automatically.
> * Configure CellDetector and Classifier automatically.
> * All config files are given to DiagnosisSystem on initialisation.

> Implement coverage check in CellDetector.

> Implement a combination of SVM and coverage check to remove false RBC detections. 

> TODO: 
> * Build and test a CNN.
> * Collect more samples.
> * Test cell detector on all new samples. 

20/06/2017
----------
> Implement a method to check if each circle given by Hough's Circle Transform contains a RBC.
> * Trained a SVM using colour histograms to test each circle.
> * Tested the method.
> * Decided that a simple coverage check may be required before hand.

> Next: refactor code. 

19/06/2017
----------
> Confirmed how data for each sample will be stored.
> * Each sample will have Image objects.
> * Each Image object will have rbc and wbc objects.

> Decided to use a pandas dataframe to store all results.

> Completed a method to let the user classify each cell manually for training.

> Completed a method to initialise the sample objects from a test.info file.
> * Each cell imaged saved for training will be labled with the sample id, image id and cell id.
> * This way, a CNN can be trained for a given set of samples.

17/06/2017
----------
> Read Prof. Dik Morling's papers on Malaria Detection.

> Read about MalaDiag.

16/06/20187
-----------
> Setup automatic backup system with versioning and synced laptop and desktop. 

15/06/2017
----------
> Understood the maing principles of Convolutional Neural Networks.

> Develop a theoretical framework for using a CNN for Malaria detection. 

12/06/2017
----------
> Down time - computer issues. 

09/06/2017
----------
> Theorised improvements to current RBC detection method.
> * A simple coverage check does not differentiat between RBCs and WBCs.
> * I will use a SVM to discard circles that are actually detecting WBCs. 

08/06/2017
----------
> Read Keras and Tensor Flow documentation

07/06/2017
----------
> Read and summarised relavent literature

> Researched and installed TensorFlow and Keras.

> Installed OpenCV.

06/06/2017
----------
> Meeting with Prof. Guido Bugmann, resultant course of action:
> Possible approaches include:
> * Machine Learning.
> * Convolutional Neural Network.
> * Machine Learning with Template Matching.
> Talk to Dr. Phil Culverhouse about his Plankton detection/classification. 
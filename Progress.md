# Project log

This is a log of any tasks completed throughout the project.

19/06/2017
----------
A sample results in a collection of images and within those images is a collection of cells.

Confirmed how data for each sample will be stored.
> Each sample will have Image objects.
> Each Image object will have rbc and wbc objects.

Decided to use a pandas dataframe to store all results.

Write a method to let the user classify each cell manually for training.

Write a method to initialise the sample objects from a test.info file.
> Each cell imaged saved for training will be labled with the sample id, image id and cell id.
> This way, a CNN can be trained for a given set of samples. 






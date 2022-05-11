# Evaluating-Explanations
6.819 Final Project: Evaluating Explanations for Training a Student to Simulate a Teacher on Image Classification

## Setup
1. Download the CIFAR-10 dataset into the ```Teacher``` directory. Instructions [here](https://www.cs.toronto.edu/~kriz/cifar.html). 
2. Create a ```models``` directory under the ```Teacher``` directory, and then run ```python3 train-teacher-model.py``` to train a teacher model, which
will be saved by default in ```Teacher/models```.
3. Create a ```data``` direcory under the ```Student``` directory, and then run ```python3 saliency-old.py``` to produce the teacher explanation augmented 
train set, as well as the test set for the student training.
4. Run ```python3 make-data.py``` to get the samples of the student train set of 1500, 3000, 4500, 6000, and 7500 (full student train set) data instances.
5. Create a ```results``` directory under ```Student```. Then create ```gradient```, ```smoothgrad```, ```middle-layer```, and ```edge-detector``` directories
under ```results```. Then, create a ```1500```, ```3000```, ```4500```, ```6000```, and ```7500``` directories under each of the above explanation directories.
Finally, run ```python3 train-student-model-multitask [explanation-type]``` to train a student model 20 times with the given explanation type at each of the
hyperparameter values discussed in the paper. The accuracy curve results will be stored in the ```results``` directory.

## Visualization, Plotting, and Computing Metrics
- Run ```python3 plot-data.py [explanation-type]``` from the ```Student``` directory to obtain the accuracy curves (averaged over the 20 trials) of each
hyperparameter value for the given explanation type.
- Run ```python3 compute-area.py [explanation-type]``` to compute the area under each of the accuracy curves for a given explanation type, as well as other
statistics such as final accuracy and fraction of area increase over baseline.

## Code References
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

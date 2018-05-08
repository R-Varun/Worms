# Worms


# Installation

## General
I have provided my frozen python environment, so after cloning the repository, you may `pip install requirements.txt` to install dependencies for the project

## Windows
For windows, you may use the provided `requirements.txt`; however, as pytorch can be difficult to install, I recommend using [Anaconda](https://conda.io/docs/user-guide/install/windows.html)



# Usage
In order to run worm detection, we use a convnet to determine whether or not contours are worms. To this end, we must either train a model, or use a pretrained model

I have included two models trained for ~ 50 Epochs, achieving 99.81%/97% accuracy Train/Test. Feel free to include your own image
(Keep in mind this model has been trained using a GPU, 

## Train model
In order to train a model, simply run `python convnet.py`. This will run some number of iterations; however, if you `ctr + c` to exit the process beforehand, you will be left with the model generated from the last epoch.

## Run Worm Detector
Now that we have a trained model, we can run our worm segmenter. The required parameters are `-input ` and `-output `. These denote the input file and output directory for the segmented videos respectively.

An example run script would be `python main.py -input 8.avi -ouput ./` to analyze a file, `8.avi` in the current working directory and outputting to the current working directory






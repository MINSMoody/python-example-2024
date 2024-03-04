#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from TeamCode.src.sample_implementation import ExampleDigitizationModel, ExampleClassificationModel

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    model = ExampleDigitizationModel() ## << We only edit this line
    model.train_model(data_folder, model_folder, verbose)


# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    model = ExampleClassificationModel() ## << We only edit this line
    model.train_model(data_folder, model_folder, verbose)

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    return ExampleDigitizationModel.from_folder(model_folder, verbose)


# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    return ExampleClassificationModel.from_folder(model_folder, verbose)
    
# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    return digitization_model.run_digitization_model(record, verbose)


# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    return  dx_model.run_dx_model(record, signal, verbose)

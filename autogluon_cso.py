#!/usr/bin/env python

import sys
import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# define the y column
label = "Cancer"

train_data = TabularDataset("files/input_training.csv")
y_train = train_data[label]  # values to predict
train_data_nolab = train_data.drop(columns=[label])  # delete label column to prove we're not cheating
train_data.head()

test_data = TabularDataset("files/input_testing.csv")
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
test_data_nolab.head()
print("Summary of class variable: \n", train_data[label].describe())

# fit model and save
save_path_ridge = "output"
predictor_ridge = TabularPredictor(label=label, path=save_path_ridge, 
                                   eval_metric="log_loss"
                                  ).fit(train_data,
                                        #presets = "best_quality",
                                        #auto_stack = True,
                                        holdout_frac = 0.2, # default
                                        time_limit = False,
                                        num_cpus = 20, num_gpus = 1)

train_pred_ridge = predictor_ridge.predict_proba(train_data_nolab)
test_pred_ridge = predictor_ridge.predict_proba(test_data_nolab)

# get model performance in training samples
if os.path.exists(save_path_ridge + "/prediction") == False:
    os.mkdir(save_path_ridge + "/prediction")
train_stats = predictor_ridge.leaderboard(
    train_data, silent = True,
    extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss']
)
train_stats.to_csv(save_path_ridge + "/prediction/Model.Stats.Training.txt",
                   na_rep = "NA",
                  sep = "\t", header = True, index = False)
# get model performance in testing samples
test_stats = predictor_ridge.leaderboard(
    test_data, silent = True,
    extra_metrics=['accuracy', 'balanced_accuracy', 'log_loss']
)
test_stats.to_csv(save_path_ridge + "/prediction/Model.Stats.Testing.txt",
                  na_rep = "NA",
                  sep = "\t", header = True, index = False)

# save predtion of individual models and ensemble model
for model_name in predictor_ridge.get_model_names():
    train_filepath = save_path_ridge + "/prediction/" + "Training." + model_name + ".prediction.txt"
    test_filepath = save_path_ridge + "/prediction/" + "Testing." + model_name + ".prediction.txt"
    
    # training samples
    train_pred = predictor_ridge.predict_proba(
        train_data_nolab, 
        model = model_name
    )
    train_pred.insert(
        0, "Sample", train_data["Sample"], 
        allow_duplicates = False
    )
    train_pred.to_csv(train_filepath, na_rep = "NA",
                      sep = "\t", header = True, index = False)
    
    # testing samples
    test_pred = predictor_ridge.predict_proba(
        test_data_nolab, 
        model = model_name
    )
    test_pred.insert(
        0, "Sample", test_data["Sample"], 
        allow_duplicates = False
    )
    test_pred.to_csv(test_filepath, na_rep = "NA",
                     sep = "\t", header = True, index = False)
    

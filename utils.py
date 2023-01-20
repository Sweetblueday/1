import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm_notebook
import torch
from datasets import load_dataset

""" Data Reader """

def LoadDatasets(DataName):
    if DataName == "TREC":
        dataset = load_dataset("trec")
        x_train = []; x_train_ids = []; y_train = []
        x_valid = []; x_valid_ids = []; y_valid = []
        x_test = []; x_test_ids = []; y_test = []
        for i in range(len(dataset["train"])):
            x_train.append(dataset["train"][i]["text"])
            y_train.append(dataset["train"][i]["coarse_label"])
        for i in range(len(dataset["test"])):
            x_test.append(dataset["test"][i]["text"])
            y_test.append(dataset["test"][i]["coarse_label"])
        TopicList = {}
        Idx2Topic =  {}
        return x_train, y_train, x_valid, y_valid, x_test, y_test, TopicList, Idx2Topic
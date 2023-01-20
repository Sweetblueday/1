# -*- coding: utf-8 -*-
import copy
import gzip
import pickle
import re
from pprint import pprint
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import json
import random
from pathlib import Path

from transformers import PreTrainedTokenizer, AutoTokenizer

def prepare_definitions(pairs=None, uncased=True):
    datasets = open("./data/DefinitionDatasetNoPOS", 'r', encoding='utf-8')
    w_d_e_dict = {}
    for line in datasets:
        line = line.split(" (def.) ")
        word = line[0].strip()
        if uncased:
            word = word.lower()
           
        defs = {}
        for l in line[1:]:
            l = l.split(" (ex.) ")
            d = l[0].strip().lower() if uncased else l[0].strip()
            e = l[1].strip().lower() if uncased else l[1].strip()
            e = e.split(' | ')
        defs[d] = e if len(e) else ''
        
        if word in w_d_e_dict.keys():
            w_d_e_dict[word].update(defs)
        else:
            w_d_e_dict[word] = defs
        
    all_data = []
    
    for word in w_d_e_dict:
        if pairs == "word-word":
            all_data.append({
                'word': word,
                'src': word,
                'tgt': word,
            })
        elif pairs == "word-def":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                all_data.append({
                    'word': word,
                    'src': word,
                    'tgt': d,
                })
        elif pairs == "def-word":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                all_data.append({
                    'word': word,
                    'src': d,
                    'tgt': word,
                })
        elif pairs == "word-exam":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                for ex in w_d_e_dict[word][d]:
                    if ex != '':
                        all_data.append({
                            'word': word,
                            'src': word,
                            'tgt': ex,
                        })
        elif pairs == "exam-word":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                for ex in w_d_e_dict[word][d]:
                    if ex != '':
                        all_data.append({
                            'word': word,
                            'src': ex,
                            'tgt': word,
                        })
        elif pairs == "def-exam":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                for ex in w_d_e_dict[word][d]:
                    if ex != '':
                        all_data.append({
                            'word': word,
                            'src': d,
                            'tgt': ex,
                        })
        elif pairs == "exam-def":
            defs = list(w_d_e_dict[word].keys())
            for d in defs:
                for ex in w_d_e_dict[word][d]:
                    if ex != '':
                        all_data.append({
                            'word': word,
                            'src': ex,
                            'tgt': d,
                        })
        else:
            print("Warning:: No Pairs")
            break
            
    return all_data

def main():
    save_dir = Path('/home/jovyan/data/def-bert')
    save_dir.mkdir(parents=True, exist_ok=True)
    ### Select dataset
    train_data = prepare_definitions('word-word')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    print(f'{save_dir}/train_data_word-word.pkl')
    with open(f'{save_dir}/train_data_word-word.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    train_data = prepare_definitions('word-def')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_word-def.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    train_data = prepare_definitions('def-word')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_def-word.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    train_data = prepare_definitions('word-exam')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_word-exam.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    train_data = prepare_definitions('exam-word')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_exam-word.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    train_data = prepare_definitions('def-exam')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_def-exam.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    train_data = prepare_definitions('exam-def')
    for i in range(len(train_data[:5])):
        print(train_data[i])
    with open(f'{save_dir}/train_data_exam-def.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
if __name__ == '__main__':
    main()

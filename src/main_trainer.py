# -*- coding: utf-8 -*-
import logging
import os
import pickle
import random
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer, AutoModel,
    Trainer, DataCollatorWithPadding, TrainingArguments,
    IntervalStrategy, EvalPrediction, PreTrainedModel,
)
from torch import distributed as dist
# from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import copy
import re
from tqdm.auto import tqdm
from collections import Counter
import string
from scipy import stats
# import gluonnlp

logger = logging.getLogger(__name__)
# ["CLS-CLS", "Target-CLS", "CLS-Target", "Target-Target"]
eval_strategy = ["CLS-CLS"]
# ["word-word", "word-def", "def-word", "word-exam", "exam-word", "def-exam", "exam-def"]
# definition_data = ["word-word", "word-def", "word-exam" ,"def-exam"]
definition_data = ["word-word"]
pair_index = {
    "word-word": 0, "word-def": 1, "def-word": 2, "word-exam": 3,
    "exam-word": 4, "def-exam": 5, "exam-def": 6,
}

def offset_indexing(tokenizer, word, text, max_length):
    input_text = f"[CLS] {text} [SEP]"
    input_text_tokenized = tokenizer(
            input_text,
            truncation=max_length is not None,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            add_special_tokens=False,
        )
    # Find All Possible Word Position
    match = list(re.finditer(re.escape(word), input_text))
    if len(match) == 0:
        return input_text, [0, 1] # Indicate [CLS]
    rand_idx = random.randint(0, len(match))
    rand_idx = 0
    for i, m in enumerate(match):
        if rand_idx == i:
            start_ch_idx = m.start()

    end_ch_idx = start_ch_idx + len(word)
    # find start
    start_token_idx = None
    for token_idx, offset in enumerate(input_text_tokenized['offset_mapping'][0]):
        start_token_idx = token_idx
        if offset[0] <= start_ch_idx < offset[1]:
            break
    # find end
    end_token_idx = None
    StopFlag = False
    for token_idx, offset in enumerate(input_text_tokenized['offset_mapping'][0]):
        end_token_idx = token_idx
        if offset[0] < end_ch_idx <= offset[1]:
            StopFlag = True
        if StopFlag and not offset[0] < end_ch_idx <= offset[1]:
            break
    target_index = [ start_token_idx, end_token_idx ] # Custom Parsing
    return input_text, target_index

class DatasetForDefinition(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, pair=None, max_length: Optional[int] = None):
        self.inputs = []
        self.targets = []
        file = '/home/jovyan/data/def-bert/train_data_'+pair+'.pkl'
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        inputs, targets = [], []
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar = tqdm(total = len(dataset))

        for data in dataset:
            word = data['word']
            src = data['src']
            tgt = data['tgt']

            input_text_src, word_index_in_src = offset_indexing(tokenizer, word, src, max_length)
            input_text_tgt, word_index_in_tgt = offset_indexing(tokenizer, word, tgt, max_length)

            inputs.append([input_text_src, input_text_tgt])
            targets.append([word_index_in_src, word_index_in_tgt])

            if dist.is_initialized() and dist.get_rank() == 0:
                pbar.update(1)
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar.close()
        
        self.inputs = inputs
        self.targets = targets

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pair = pair

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sent1, sent2 = self.inputs[index]
        sent1_tokenized = self.tokenizer(
            sent1,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length"
        )
        
        sent2_tokenized = self.tokenizer(
            sent2,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length"
        )

        return {
            "input_ids": sent1_tokenized.input_ids,
            "input_ids_": sent2_tokenized.input_ids,
#             "labels": None,
            "word_index": self.targets[index],
            "data_pair": pair_index[self.pair],
        }

class DatasetForWordSimilarity(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, data_name, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
#         if data_name == "wordsim-sim":
#             self.data = gluonnlp.data.WordSim353('similarity', root='./data/wordsim353')
#         elif data_name == "wordsim-rel":
#             self.data = gluonnlp.data.WordSim353('relatedness', root="./data/wordsim353")
#         elif data_name == "men":
#             self.data = gluonnlp.data.MEN(root="./data/men")
#         rw = gluonnlp.data.RareWords()
#         men = gluonnlp.data.MEN()
#         simlex = gluonnlp.data.SimLex999()

        simverb = []
        file_object = open("./data/SimVerb-3000-test.txt", 'r')
        for line in file_object:
            line = line.split()
            simverb.append([line[0], line[1], float(line[3])])
        file_object.close()
#         bakerverb = gluonnlp.data.BakerVerb143()
#         yangverb = gluonnlp.data.YangPowersVerb130()
#         inputs, targets = [], []
        self.data = simverb
    
        inputs, targets, scores = [], [], []
        for data in self.data:
            input_text_src, word_index_in_src = offset_indexing(tokenizer, data[0], data[0], max_length)
            input_text_tgt, word_index_in_tgt = offset_indexing(tokenizer, data[1], data[1], max_length)
            
            inputs.append([input_text_src, input_text_tgt])
            targets.append([word_index_in_src, word_index_in_tgt])
            scores.append(data[2])
            
        self.inputs = inputs
        self.targets = targets
        self.scores = scores
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
#         w1, w2, score = self.data[index]
        w1, w2 = self.inputs[index]
        w1_tokenized = self.tokenizer(
            w1,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length"
        )
        
        w2_tokenized = self.tokenizer(
            w2,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length"
        )
        
        return {
            "input_ids": w1_tokenized.input_ids,
            "input_ids_": w2_tokenized.input_ids,
            "labels": self.scores[index],
            "word_index": self.targets[index],
        }
        
class ComputeMetricsForLogits(object):
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        
    def word_similarity(self, model_scores, human_scores): # By Hwiyeol
        return np.round(stats.spearmanr(model_scores, human_scores)[0], 3)
    
    def __call__(self, p: EvalPrediction):
        # loss , left , right, scores
        l_emb, r_emb, word_index = p[0]
        human_score = p[1]
        model_scores = []
        for b in range(len(l_emb)):
            ### CLS to CLS
            if "CLS-CLS" in eval_strategy:
                model_scores.append(cosine_similarity(l_emb[b,0:1,:], r_emb[b])[0][0])
            elif "Target-CLS" in eval_strategy:
                model_scores.append(cosine_similarity(
                    l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(axis=0, keepdims=True),
                    r_emb[b,0:1,:])[0][0]
                )
            elif "CLS-Target" in eval_strategy:
                model_scores.append(cosine_similarity(
                    l_emb[b,0:1,:],
                    r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(axis=0, keepdims=True))[0][0]
                )
            elif "Target-Target" in eval_strategy:
                model_scores.append(cosine_similarity(
                    l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(axis=0, keepdims=True),
                    r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(axis=0, keepdims=True))[0][0]
                )
                
        return {
#             'em_acc': np.round(1, 3),
            'word_sim_corr': self.word_similarity(model_scores, human_score),
        }

class Model(PreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        self.config = config
        self.model_train = AutoModel.from_pretrained(model_name)
        self.model_ref = AutoModel.from_pretrained(model_name)
        self.model_ref.eval()
        self.loss_func = nn.MSELoss()
#         self.cnn_block = CnnBlock(1024, 1024, kernel_size=3, num_block=5)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        word_index=None,
        data_pair=None,
        input_ids_=None, # auxiliary used
    ):          
        if labels is None: # train; uses the fixed one
            outputs_train = self.model_train(
                input_ids=input_ids,
                attention_mask=torch.where(input_ids != 0, 1, 0),
                token_type_ids=torch.zeros_like(input_ids),
            )
                
            with torch.no_grad():
                outputs_ref = self.model_ref(
                    input_ids=input_ids_,
                    attention_mask=torch.where(input_ids_ != 0, 1, 0),
                    token_type_ids=torch.zeros_like(input_ids_),
                )
            
        else: # prediction; uses the same trained one
            with torch.no_grad():
                outputs_train = self.model_train(
                    input_ids=input_ids,
                    attention_mask=torch.where(input_ids != 0, 1, 0),
                    token_type_ids=torch.zeros_like(input_ids),
                )
                
                outputs_train_ = self.model_train(
                    input_ids=input_ids_,
                    attention_mask=torch.where(input_ids_ != 0, 1, 0),
                    token_type_ids=torch.zeros_like(input_ids_),
                )
                
        if labels is None:
            l_emb, r_emb = outputs_train["last_hidden_state"], outputs_ref["last_hidden_state"]
        else:
            l_emb, r_emb = outputs_train["last_hidden_state"], outputs_train_["last_hidden_state"]
        ##### Loss Variations
        # ["word-word", "word-def", "def-word", "word-exam", "exam-word", "def-exam", "exam-def"]
        loss = 0
        ### CLS to CLS
        if data_pair is not None:
            if data_pair[0] in [pair_index["word-def"], pair_index["def-word"]]:
                loss += self.loss_func(l_emb[:,0,:], r_emb[:,0,:].detach())
            ### Target to CLS; word_index[batch, left/right, start/end]
            if data_pair[0] in [pair_index["exam-word"], pair_index["exam-def"]]:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    r_emb[:,0,:].detach()
                )
            ### CLS to Target
            if data_pair[0] in [pair_index["word-word"], pair_index["word-exam"], pair_index["def-exam"]]:
                loss += self.loss_func(
                    l_emb[:,0,:],
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
            ### Target to Target
            if data_pair[0] in []:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )

        else:
            if "CLS-CLS" in eval_strategy:
                loss += self.loss_func(l_emb[:,0,:], r_emb[:,0,:].detach())
            elif "Target-CLS" in eval_strategy:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    r_emb[:,0,:].detach()
                )
            elif "CLS-Target" in eval_strategy:
                loss += self.loss_func(
                    l_emb[:,0,:],
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
            elif "Target-Target" in eval_strategy:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
        
        return loss, l_emb, r_emb, word_index
        

def train():
    local_rank = os.getenv("LOCAL_RANK")
    
    args = TrainingArguments(
        output_dir='~/ckpt/def-bert/',
        overwrite_output_dir=True,
        logging_first_step=True,
        do_train=True,
        do_eval=True,
        learning_rate=1e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        logging_steps=1000,
        seed=42,
        local_rank=int(local_rank) if local_rank is not None else -1,
        save_strategy=IntervalStrategy.EPOCH,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transformer_model = AutoModel.from_pretrained("bert-base-uncased")
#     for data_type in definition_data[1:]:
#         train_data = torch.utils.data.ConcatDataset(
#                         [train_data,
#                         DatasetForDefinition(tokenizer, data_type, max_length=256)]
#         )

    for def_data in definition_data:
        model: Model = Model(transformer_model.config, "bert-base-uncased")
        train_data = DatasetForDefinition(tokenizer, def_data, max_length=256)        
        eval_data = DatasetForWordSimilarity(tokenizer, 'simverb', max_length=256)
        metrics = ComputeMetricsForLogits(tokenizer)
        trainer = Trainer(
            model,
            args,
    #         DataCollatorWithPadding(tokenizer, padding='max_length', max_length=512),
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            compute_metrics=metrics,
        )
        
        trainer.train()

if __name__ == '__main__':
    fire.Fire({
        'train': train,
#         'test': test,
    })

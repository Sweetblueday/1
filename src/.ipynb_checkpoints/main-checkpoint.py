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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer, AutoModel,
    Trainer, DataCollatorWithPadding, TrainingArguments,
    IntervalStrategy, EvalPrediction, PreTrainedModel,
)
from torch import distributed as dist
import copy
import re
from tqdm.auto import tqdm
from collections import Counter
import string
from scipy import stats
# import gluonnlp

from accelerate import Accelerator, DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

logger = logging.getLogger(__name__)
# ["CLS-CLS", "Target-CLS", "CLS-Target", "Target-Target"]
eval_strategy = ["CLS-CLS"]
# ["word-word", "word-def", "def-word", "word-exam", "exam-word", "def-exam", "exam-def"]
# definition_data = ["word-exam", "word-word", "word-def","def-exam"]
# definition_data = ["def-exam", "exam-def"]
# definition_data = ["exam-def", "def-exam"]
definition_data = ["word-word", "exam-def"]

pair_index = {
    "word-word": 0, "word-def": 1, "def-word": 2, "word-exam": 3,
    "exam-word": 4, "def-exam": 5, "exam-def": 6,
}

BATCH_SIZE = 2**7
LEARNING_RATE = 2e-5
EPOCH = 20

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
    def __init__(self, tokenizer: PreTrainedTokenizerFast, pair=None, max_length: Optional[int] = None, dataset_for = "train"):
        self.inputs = []
        self.targets = []
        file = '/home/jovyan/data/def-bert/train_data_'+pair+'.pkl'
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        inputs, targets = [], []
        
        random.seed(42)
        random.shuffle(dataset)
        
        if dataset_for == "train":
            dataset = dataset[:int(0.8*len(dataset))]
        else: # dataset_for validation
            dataset = dataset[int(0.8*len(dataset)):]
        
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
            padding="max_length",
            return_tensors="pt",
        )
        
        sent2_tokenized = self.tokenizer(
            sent2,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

#         return sent1_tokenized.input_ids, sent2_tokenized.input_ids,\
#                     self.targets[index], pair_index[self.pair]
        return {
            "input_ids": sent1_tokenized.input_ids[0],
            "input_ids_": sent2_tokenized.input_ids[0],
#             "labels": None,
            "word_index": torch.tensor(self.targets[index]),
            "data_pair": torch.tensor(pair_index[self.pair]),
        }

class DatasetForWordSimilarity(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, data_name, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if data_name == "wordsim-sim" or data_name == "all":
            wordsim_sim = []
            file_object = open("./wordsim_data/wordsim353/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt", 'r')
            for line in file_object:
                line = line.split()
                wordsim_sim.append([line[0], line[1], float(line[2])])
            file_object.close()
            self.data += wordsim_sim
            
        if data_name == "wordsim-rel" or data_name == "all":
            wordsim_rel = []
            file_object = open("./wordsim_data/wordsim353/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt", 'r')
            for line in file_object:
                line = line.split()
                wordsim_rel.append([line[0], line[1], float(line[2])])
            file_object.close()
            self.data += wordsim_rel
        
        if data_name == "rw" or data_name == "all":
            rw = []
            file_object = open("./wordsim_data/rw/rw.txt", 'r')
            for line in file_object:
                line = line.split()
                rw.append([line[0], line[1], float(line[2])])
            file_object.close()
            self.data += rw
            
        if data_name == "men" or data_name == "all":
            men = []
            file_object = open("./wordsim_data/men/MEN/MEN_dataset_natural_form_full", 'r')
            for line in file_object:
                line = line.split()
                men.append([line[0], line[1], float(line[2])])
            file_object.close()
            self.data += men
            
        if data_name == "sem" or data_name == "all":
            sem = []
            file_object1 = open("./wordsim_data/SemEval17-Task2/test/subtask1-monolingual/data/en.test.data.txt", 'r')
            file_object2 = open("./wordsim_data/SemEval17-Task2/test/subtask1-monolingual/keys/en.test.gold.txt", 'r')
            for line1, line2 in zip(file_object1, file_object2):
                line1 = line1.split()
                line2 = line2.split()
                sem.append([line1[0], line1[1], float(line2[0])])
            file_object1.close()
            file_object2.close()
            self.data += sem
        
        if data_name == "simlex" or data_name == "all":
            simlex = []
            file_object = open("./wordsim_data/SimLex-999/SimLex-999.txt", 'r')
            file_object.readline() # skip the first line
            for line in file_object:
                line = line.split()
                simlex.append([line[0], line[1], float(line[3])])
            file_object.close()
            self.data += simlex
            
        if data_name == "simverb" or data_name == "all":
            simverb = []
            file_object = open("./wordsim_data/SimVerb-3000-test.txt", 'r')
            for line in file_object:
                line = line.split()
                simverb.append([line[0], line[1], float(line[3])])
            file_object.close()
            self.data += simverb
    
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
            padding="max_length",
            return_tensors="pt",
        )
        
        w2_tokenized = self.tokenizer(
            w2,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": w1_tokenized.input_ids[0],
            "input_ids_": w2_tokenized.input_ids[0],
            "labels": torch.tensor(self.scores[index]),
            "word_index": torch.tensor(self.targets[index]),
        }
        
# class ComputeMetricsForLogits(object):
#     def __init__(self, tokenizer: PreTrainedTokenizerFast):
#         self.tokenizer = tokenizer
        
#     def word_similarity(self, model_scores, human_scores): # By Hwiyeol
#         return np.round(stats.spearmanr(model_scores, human_scores)[0], 3)
    
#     def __call__(self, p: EvalPrediction):
#         # loss , left , right, scores
#         l_emb, r_emb, word_index = p[0]
#         human_score = p[1]
#         model_scores = []
#         for b in range(len(l_emb)):
#             ### CLS to CLS
#             if "CLS-CLS" in eval_strategy:
#                 model_scores.append(cosine_similarity(l_emb[b], r_emb[b])[0][0])
#             elif "Target-CLS" in eval_strategy:
#                 model_scores.append(cosine_similarity(
#                     l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(axis=0, keepdims=True),
#                     r_emb[b])[0][0]
#                 )
#             elif "CLS-Target" in eval_strategy:
#                 model_scores.append(cosine_similarity(
#                     l_emb[b],
#                     r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(axis=0, keepdims=True))[0][0]
#                 )
#             elif "Target-Target" in eval_strategy:
#                 model_scores.append(cosine_similarity(
#                     l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(axis=0, keepdims=True),
#                     r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(axis=0, keepdims=True))[0][0]
#                 )
                
#         return {
# #             'em_acc': np.round(1, 3),
#             'word_sim_corr': self.word_similarity(model_scores, human_score),
#         }

class Model(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.config = config
        self.model_train = model
        self.model_train.requires_grad_(True) # Line 2
        self.model_ref = copy.deepcopy(model)
#         self.model_ref = AutoModel.from_pretrained("bert-base-uncased")
        self.model_ref.requires_grad_(False)
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
            l_emb, r_emb = outputs_train["last_hidden_state"], outputs_ref["last_hidden_state"].detach()
        else:
            l_emb, r_emb = outputs_train["last_hidden_state"], outputs_train_["last_hidden_state"].detach()
        ##### Loss Variations
        # ["word-word", "word-def", "def-word", "word-exam", "exam-word", "def-exam", "exam-def"]
        loss = 0
        ### CLS to CLS
        if data_pair is not None:
            if data_pair[0] in [pair_index["word-def"], pair_index["def-word"]]:
                loss += self.loss_func(l_emb[:,0,:], r_emb[:,0,:].detach())
            ### Target to CLS; word_index[batch, left/right, start/end]
            if data_pair[0] in [pair_index["word-word"], pair_index["exam-word"], pair_index["exam-def"]]:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    r_emb[:,0,:].detach()
                )
                loss = loss/BATCH_SIZE
            ### CLS to Target
            if data_pair[0] in [pair_index["word-word"], pair_index["word-exam"], pair_index["def-exam"]]:
                loss += self.loss_func(
                    l_emb[:,0,:],
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
                loss = loss/BATCH_SIZE
            ### Target to Target
            if data_pair[0] in []:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
                loss = loss/BATCH_SIZE

        else:
            if "CLS-CLS" in eval_strategy:
                loss += self.loss_func(l_emb[:,0,:], r_emb[:,0,:].detach())
            elif "Target-CLS" in eval_strategy:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    r_emb[:,0,:].detach()
                )
                loss = loss/BATCH_SIZE
            elif "CLS-Target" in eval_strategy:
                loss += self.loss_func(
                    l_emb[:,0,:],
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
                loss = loss/BATCH_SIZE
            elif "Target-Target" in eval_strategy:
                loss += self.loss_func(
                    torch.cat(( [ l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True) for b in range(l_emb.size(0)) ]), dim=0),
                    torch.cat(( [ r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True) for b in range(r_emb.size(0)) ]), dim=0).detach()
                )
                loss = loss/BATCH_SIZE
        
        return loss, l_emb, r_emb, word_index
        

def train():
    local_rank = os.getenv("LOCAL_RANK")
#     gpu_count = torch.cuda.device_count()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs('/home/jovyan/temp/def-bert/s1/', exist_ok=True)

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained('/home/jovyan/temp/def-bert/s1/')
    model_from_transformer = AutoModel.from_pretrained("bert-base-uncased")
    model: Model = Model(model_from_transformer.config, model_from_transformer) # Wrapper
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        
    for def_data in definition_data: # definition datatype loop
        train_data = DatasetForDefinition(tokenizer, def_data, max_length=256, dataset_for="train")
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
        
        valid_data = DatasetForDefinition(tokenizer, def_data, max_length=256, dataset_for="validation")
        valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
        
        word_sim_data = DatasetForWordSimilarity(tokenizer, 'all', max_length=256)
        word_sim_loader = DataLoader(word_sim_data, batch_size=BATCH_SIZE)
        
        model, optimizer, train_loader, valid_loader, word_sim_loader =\
            accelerator.prepare(model, optimizer, train_loader, valid_loader, word_sim_loader)

        if accelerator.is_main_process:
            print('='*10)
            print(def_data)
            print('='*10)
            
        ep_best = 0
#         ep_pbar = tqdm(total = EPOCH,
#                     disable=not accelerator.is_main_process,
#                     desc=f'Train')
        for ep in range(1,EPOCH+1):
            ## training_loop
            model.train()
            total_loss = 0
            batch_pbar = tqdm(total=len(train_loader),
                        disable=not accelerator.is_main_process,
                        desc=f'train epoch {ep}')
            for batch in train_loader:
                loss, l_emb, r_emb, word_index = model(**batch)
        #         loss = loss_function(outputs, targets)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item()
                batch_pbar.update(1)
            batch_pbar.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"\n>>> train_loss : {np.round(total_loss, 3)}\n")

            total_loss = 0
            min_loss = 10**5
            model.eval()
            batch_pbar = tqdm(total=len(valid_loader),
                        disable=not accelerator.is_main_process,
                        desc=f'valid epoch {ep}')
            for batch in valid_loader:
                with torch.no_grad():
                    loss, l_emb, r_emb, word_index = model(**batch)
                total_loss += loss.item()
                batch_pbar.update(1)
            batch_pbar.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"\n>>> valid_loss : {np.round(total_loss, 3)}\n")
                if total_loss < min_loss: # best checkpoint
                    min_loss = total_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    ep_best = ep
                    ### Model save (if possible ...)
                    while True:
                        try:
                            accelerator.save(unwrapped_model.state_dict(), '/home/jovyan/temp/def-bert/s1/model.pt')
                            break
                        except OSError:
                            print("save failed")
                            pass # continue

            ## word_sim_corr evaluation
            if ep_best == ep:
                model.eval()
                model_scores, human_scores = [], []
                batch_pbar = tqdm(total=len(word_sim_loader),
                            disable=not accelerator.is_main_process,
                            desc=f'word_sim_corr epoch {ep}')
                for batch in word_sim_loader:
                    human_scores += batch['labels'].cpu().tolist()
                    with torch.no_grad():
                        loss, l_emb, r_emb, word_index = model(**batch)
                        total_loss += loss.item()
                    for b in range(batch['labels'].size(0)):
                        if "CLS-CLS" in eval_strategy:
                            model_scores += nn.functional.cosine_similarity(l_emb[b,0:1,:], r_emb[b,0:1,:]).cpu().tolist()
                        elif "Target-CLS" in eval_strategy:
                            model_scores += nn.functional.cosine_similarity(
                                l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True),
                                r_emb[b,0:1,:]).cpu().tolist()
                        elif "CLS-Target" in eval_strategy:
                            model_scores += nn.functional.cosine_similarity(
                                l_emb[b,0:1,:],
                                r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True)).cpu().tolist()
                        elif "Target-Target" in eval_strategy:
                            model_scores += nn.functional.cosine_similarity(
                                l_emb[b,word_index[b,0,0]:word_index[b,0,1],:].mean(dim=0, keepdim=True),
                                r_emb[b,word_index[b,1,0]:word_index[b,1,1],:].mean(dim=0, keepdim=True)).cpu().tolist()
                    batch_pbar.update(1)
            if accelerator.is_main_process:
#                 ep_pbar.update(1)
                batch_pbar.close()
                word_sim_corr = np.round(stats.spearmanr(model_scores, human_scores)[0], 3)
                print(f"\n>>> word_sim_corr @ ep {ep}: {word_sim_corr}\n") 
#         ep_pbar.close()
        if accelerator.is_main_process:
            print(f">>> best @ ep {ep_best}: {word_sim_corr}")
            
if __name__ == '__main__':
    fire.Fire({
        'train': train,
#         'test': test,
    })
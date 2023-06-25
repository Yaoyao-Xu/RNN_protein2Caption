import torch
import spacy
import numpy as np
import random
import math
import torchtext
from collections import Counter

spacy_en = spacy.load('en_core_web_sm')
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data:
    def __init__(self, sequence, function,sequence_eva, function_eva):
        # 读取数据 并分词
        self.protein, self.fun = self.get_token(sequence, function)
        self.protein_eva, self.fun_eva = self.get_token(sequence_eva, function_eva)
        # 构建单词表
        self.word2idx_src, self.src_len, self.idx2word_src = self.build_dict(self.protein, max_words=50000)
        self.word2idx_trg, self.trg_len, self.idx2word_trg = self.build_dict(self.fun, max_words=500000)

        self.out_s, self.out_t = self.wordToID(self.protein, self.fun, self.word2idx_src, self.word2idx_trg, sort=True)
        self.out_s_eva, self.out_t_eva = self.wordToID(self.protein_eva, self.fun_eva, self.word2idx_src, self.word2idx_trg, sort=True)
        
        self.maxLength_src_train = max(len(x) for x in self.out_s)
        self.maxLength_trg_train = max(len(y) for y in self.out_t)
        self.maxLength_src_eva = max(len(x) for x in self.out_s_eva)
        self.maxLength_trg_eva = max(len(y) for y in self.out_t_eva)

        self.maxLength_src = max(self.maxLength_src_train, self.maxLength_src_eva)
        self.maxLength_trg = max(self.maxLength_trg_train, self.maxLength_trg_eva)


        self.src_pad = self.padding(self.out_s,self.maxLength_src )
        self.trg_pad = self.padding(self.out_t,self.maxLength_trg)
        self.src_eva_pad = self.padding(self.out_s_eva,self.maxLength_src)
        self.trg_eva_pad = self.padding(self.out_t_eva,self.maxLength_trg)
        self.src, self.trg = self.data_to_tensor(self.src_pad, self.trg_pad)
        self.src_eva, self.trg_eva = self.data_to_tensor(self.src_eva_pad, self.trg_eva_pad)
        
    
    def get_token(self, sequence, function):
        protein =[]
        fun =[]
        for  x in sequence:
            a = [tok.text for tok in spacy_en.tokenizer(x)]
            protein.append(a)
        for y in function:
            b = [tok.text for tok in spacy_en.tokenizer(y)]
            fun.append(b)
        return protein, fun
    
    def build_dict(self, sentences, max_words=5000):
        
        # count the words in the dataset
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = 1 #unknown token
        word_dict['PAD'] = 0 #pad token to reach the longest token size of the sample in the training set

        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict
    
    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        
        length = len(en)
        # change the src and trg from the token to idx through the vocab built
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # sort function
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # sort the src and trg based on the order
        if sort:

            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids
    
    
    def padding(self, data, max_word_length):
        for x in data:
            while len(x)<max_word_length:
                x.append(0)
        return data

    def data_to_tensor(self, input, groud_truth):
        input = torch.tensor(input) ## change the data shape to match the network input 
        groud_truth = torch.tensor(groud_truth)  #
        input = torch.transpose(input, 1,0)
        groud_truth = torch.transpose(groud_truth, 1,0)
        input = input.to(device)
        groud_truth = groud_truth.to(device)

        return input, groud_truth
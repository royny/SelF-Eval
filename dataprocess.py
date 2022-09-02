import json
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import RobertaTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import copy
import torch
import math
from tqdm import tqdm
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle as pkl

class Leveled_dataset(Dataset):
    def __init__(self, file_path, split2four=False, level='medium', mode='train', use_bleu=False):
        '''
            three levels of data. In fact, can be 4.
        '''
        self.data_path = file_path
        self.mode = mode
        self.raw_data, self.turn_len = self._data_turn_sort()
        self.split2four = split2four
        self.use_bleu = use_bleu
        self.dialog_concat()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.level = level
        #self.classify_data = []

    def _data_turn_sort(self):
        '''
            raw_data : lists of dict {'context':[], 'positive_responses':[]}
            turn_sorted_index = indice sorted by length of turns in dialog
        '''
        if self.mode == 'train':
            
            with open(self.data_path+'/train.json', 'r') as f:
                data1 = [json.loads(i.strip()) for i in f.readlines()]
            '''
            with open(self.data_path+'/dev.json', 'r') as f:
                data2 = [json.loads(i.strip()) for i in f.readlines()]
            
            with open(self.data_path+'/test.json', 'r') as f:
                data3 = [json.loads(i.strip()) for i in f.readlines()]
            '''
            data = data1[:10]#+data3+data2

        else:
            with open(self.data_path, 'r') as f:
                data = [json.loads(i.strip()) for i in f.readlines()]
        '''
        turn_len = []
        for i in data:
            turn_len.append(len(i['context'])+1)
        '''
        turn_len = [len(i['context'])+1 for i in data]
        
        len_slot = set(turn_len)

        turn_sorted_index = {key:[] for key in len_slot}

        for num, i in enumerate(turn_len):
            turn_sorted_index[i].append(num)
        
        return data, turn_sorted_index

    def dialog_concat(self):
        #problem: when spliting to 4, dialog length longer than 4 will produce cluster whose 5 dialogs are all the same.
        data = []
        indice = []
        if self.split2four:
            cur_len = 4
            while self.turn_len.get(cur_len):
                for i in self.turn_len[cur_len]:
                    cur_data = [self.raw_data[i]['context']+[pr] for pr in self.raw_data[i]['positive_responses']]
                    for j in range(cur_len-4+1):
                        mid = [k[j:4+j] for k in cur_data]
                        cluster = (mid, cur_len-4)
                        data.append(cluster)
                cur_len += 1
        else:
            cur_len = 4
            while self.turn_len.get(cur_len):
                for i in self.turn_len[cur_len]:
                    cur_data = [self.raw_data[i]['context']+[pr] for pr in self.raw_data[i]['positive_responses']]
                    data.append(cur_data)
                indice.append(len(data))
                cur_len += 1
                

        self.indice = indice
        self.cur_data = data

    def __len__(self):
        if self.level == 'medium':
            return len(self.cur_data)
        elif self.level == 'easy':
            return 5*len(self.cur_data)
        else:
            return 10000

    def __getitem__(self, idx):
        batch = []
        if self.level == 'medium':
            pair1 = self.cur_data[idx] # change it for list, if tuple, self.cur_data[idx][0] 
        elif self.level == 'easy':
            remainder = idx % 5
            num_five = int(idx/5)
            pair1 = []
            for i in range(5):
                pair1.append(copy.copy(self.cur_data[num_five][remainder]))
        elif self.level == 'hard':
            select_len = 0
            idx_in_cur_data = int((idx/10000)*self.indice[-1])
            for num, i in enumerate(self.indice):
                if idx_in_cur_data < i:
                    select_len += num
                    break
            if select_len:
                rand = torch.randint(self.indice[select_len-1], self.indice[select_len], (5,))
            else:
                rand = torch.randint(0, self.indice[select_len], (5,))
            pair1 = []
            for i,j in enumerate(rand):
                pair1.append(self.cur_data[j][i])

        if self.use_bleu:
            pairs = self.create_bleu_neg_samples(pair1, idx)
        else:
            pairs = self.create_neg_samples(pair1, idx)

        if self.split2four:
            batch.extend(self.concat_with_sep(pair1[0])) #0 when split 2 four,just pair1 otherwise.
        else:
            batch.extend(self.concat_with_sep(pair1))

        if isinstance(pairs, tuple):
            for k in pairs:
                batch.extend(self.concat_with_sep(k))
        else:
            batch.extend(self.concat_with_sep(pairs))
        #print(pair1,len(pair1))
        #print(batch, len(batch))
        ts_batch = self.tokenizer(batch, padding='max_length', truncation=True, return_tensors="pt") #padding='max_length'
        #a = self.tokenizer(' </s></s> ')
        #print(ts_batch['input_ids'].size())
        #print(a['input_ids'])
        #print(ts_batch['input_ids'][0])
        #print(self.tokenizer.decode(ts_batch['input_ids'][0]))
        ts_batch = {i: j.view(-1,5,512) for i, j in ts_batch.items()}

        ts_sep = torch.ones(3,5,200)
        for n1,i in enumerate(ts_batch['input_ids']):
            for n2,j in enumerate(i):
                t_len = 0
                for n3,k in enumerate(j):
                    if k == 2:
                        ts_sep[n1][n2][t_len] = n3
                        t_len += 1

        #print(ts_sep[0][0])
        #print(ts_batch['input_ids'][0][0][:80])
        #print(self.tokenizer.decode(ts_batch['input_ids'][0][0][:80]))
        return ts_batch, ts_sep#self.classify_data 

    def create_neg_samples(self, pos, idx):
        '''
        pos, tuple(for fixed len4 size of inputs), list of positive dialog and its turn length
        lists (dialog cluster(5 dialogs),cur_len) turn_len = 4 since all of them are splited into 4-turn dialog
        cur_len denotes the initial length minus 4 of the dialog before splited
        '''
        if isinstance(pos, tuple):
            '''
                What we need to decide is the balance between times of random sampling and 
                the size of candidate pool.
                This way the candidate pool is rather small.
            '''
            pair2 = []
            pair3 = []

            rand1_tar = np.random.randint(4, size=5)
            rand1_repl = np.random.randint(len(self.cur_data))
            '''
            #try a bigger candidate pool
            rand1_rep = np.random.randint(len(self.cur_data), size = 5)
            while True in [idx+i in rand1_rep for i in range(-pos[1], pos[1]+1)]:
                rand1_rep = np.random.randint(len(self.cur_data), size = 5)
            '''
            wrong_samples = [i+idx for i in range(-pos[1], pos[1]+1)]
            while rand1_repl in wrong_samples:
                rand1_repl = np.random.randint(len(self.cur_data))
            
            pair2 = copy.deepcopy(pos[0])
            for i,j in zip(range(len(pos[0])), rand1_tar):
                pair2[i][j] = self.cur_data[rand1_repl][0][i][j]


            #rand2_tar = np.random.randint(2, size=5)
            rand2_repl = np.random.randint(len(self.cur_data))
            while rand2_repl in wrong_samples:
                rand2_repl = np.random.randint(len(self.cur_data))

            pair3 = copy.deepcopy(pos[0])
            for i,j in zip(range(len(pos[0])), rand1_tar):
                if j%2 == 0:
                    pair3[i][0] = self.cur_data[rand2_repl][0][i][0]
                    pair3[i][2] = self.cur_data[rand2_repl][0][i][2]
                else:
                    pair3[i][1] = self.cur_data[rand2_repl][0][i][1]
                    pair3[i][3] = self.cur_data[rand2_repl][0][i][3]

            return pair2, pair3
        else:
            pair2 = []
            pair3 = []
            dialog_len = len(pos[0])
            cluster_len = len(pos) #5
            # disturb 1 turn
            rand1_tar = torch.randint(dialog_len, (5,))
            rand1_repl = torch.randint(len(self.cur_data), (5,))
            pair2 = copy.deepcopy(pos)
            
            
            for i, j, k in zip(range(cluster_len), rand1_tar, rand1_repl):
                #pair2[i][j] = self.cur_data[k][i][torch.randint(len(self.cur_data[k][0]), (1,))] don't need that much possibility currently
                pair2[i][j] = self.cur_data[k][i][0]
            if dialog_len < 4:
                return pair2    
            
            # disturb two turns
            rand2_repl = torch.randint(len(self.cur_data), (5,2))
            pair3 = copy.deepcopy(pos)
            
            for i,j in zip(range(cluster_len), rand2_repl):
                rand2_tar0 = torch.randint(2,(1,))
                if dialog_len % 2:
                    binary = math.floor(dialog_len/2) + int(rand2_tar0)
                else:
                    binary = math.floor(dialog_len/2)

                rand2_tar1 = torch.randint(binary,(2,))
                while rand2_tar1[0] == rand2_tar1[1]:
                    rand2_tar1 = torch.randint(binary,(2,))
                pair3[i][2*rand2_tar1[0]+(1-int(rand2_tar0))] = self.cur_data[j[0]][i][0]
                pair3[i][2*rand2_tar1[1]+(1-int(rand2_tar0))] = self.cur_data[j[1]][i][0]
            
            #if dialog_len < 6:
            return pair2, pair3

            '''
            pair4 = copy.deepcopy(pos)
            rand3_repl = torch.randint(len(self.cur_data), (5,3))
            rand3_tar = torch.randint(2, (5,))
            for i,j in zip(range(cluster_len), rand3_repl):
                if rand3_tar[i] % 2 == 1:
                    for num, k in enumerate(range(1, dialog_len, 2)):
                        pair4[i][k] = self.cur_data[j[num]][i][0]
                else:
                    if dialog_len == 7:
                        maintain = torch.randint(4, (1,))
                        num = 0
                        for k in range(0, dialog_len, 2):
                            if k == 2*maintain: continue
                            pair4[i][k] = self.cur_data[j[num]][i][0]
                            num += 1
                    else:
                        for num, k in enumerate(range(0, dialog_len, 2)):
                            pair4[i][k] = self.cur_data[j[num]][i][0]
            return pair2, pair3, pair4
            '''
                    


    def create_bleu_neg_samples(self, pos, idx):
        if isinstance(pos, tuple):
            '''
                What we need to decide is the balance between times of random sampling and 
                the size of candidate pool.
                This way the candidate pool is rather small.
            '''
            pair2 = []
            pair3 = []
            rand1_tar = np.random.randint(4, size=5)
            rand1_repl = np.random.randint(len(self.cur_data), size=10)
            '''
            #try a bigger candidate pool
            rand1_rep = np.random.randint(len(self.cur_data), size = 5)
            while True in [idx+i in rand1_rep for i in range(-pos[1], pos[1]+1)]:
                rand1_rep = np.random.randint(len(self.cur_data), size = 5)
            '''
            wrong_samples = [i+idx for i in range(-pos[1], pos[1]+1)]
            
            while rand1_repl.all() in wrong_samples:
                rand1_repl = np.random.randint(len(self.cur_data), size=10)
            
            pair2 = copy.deepcopy(pos[0])
            mode_me = True
            if pair2[0][3] == pair2[1][3]:
                mode_me = False
            used_sen = []
            for i,j in zip(range(len(pos[0])), rand1_tar):
                ref = self.create_ref(pair2, mode_me, i, j)
                tar = [self.cur_data[k][0][i][j] for k in rand1_repl]
                best_idx = self.get_idx(ref,tar,used_idx=False,used_sen=used_sen)
                used_sen.append(tar[best_idx])
                pair2[i][j] = tar[best_idx]


            #rand2_tar = np.random.randint(2, size=5)
            rand2_repl = np.random.randint(len(self.cur_data), size=10)
            while rand2_repl.all() in wrong_samples:
                rand2_repl = np.random.randint(len(self.cur_data))

            pair3 = copy.deepcopy(pos[0])
            used_idx = [[], [], [], []]
            tar0 = [self.cur_data[k][0][0][0] for k in rand1_repl]
            tar1 = [self.cur_data[k][0][0][1] for k in rand1_repl]
            tar2 = [self.cur_data[k][0][0][2] for k in rand1_repl]
            for i,j in zip(range(len(pos[0])), rand1_tar):
                if j%2 == 0:
                    ref1 = [[pair2[i][0]], [pair2[i][1]]]
                    ref2 = self.create_ref(pair2, mode_me, i, 2)
                    best_idx1 = self.get_idx(ref1, tar0, used_idx[0])
                    best_idx2 = self.get_idx(ref2, tar2, used_idx[2])
                    used_idx[0].append(best_idx1)
                    used_idx[2].append(best_idx2)
                    pair3[i][0] = tar0[best_idx1]
                    pair3[i][2] = tar2[best_idx2]
                else:
                    ref1 = [[pair2[i][0]], [pair2[i][1]], [pair2[i][2]]]
                    ref2 = self.create_ref(pair2, mode_me, i, 3)
                    tar3 = [self.cur_data[k][0][i][3] for k in rand1_repl]
                    best_idx1 = self.get_idx(ref1, tar1, used_idx[1])
                    best_idx2 = self.get_idx(ref2, tar3, used_idx[3])
                    used_idx[1].append(best_idx1)
                    used_idx[3].append(best_idx2)
                    pair3[i][1] = tar1[best_idx1]
                    pair3[i][3] = tar3[best_idx2]

            return pair2, pair3
        else:
            pair2 = []
            pair3 = []
            dialog_len = len(pos[0])
            cluster_len = len(pos) #5
            # disturb 1 turn
            rand1_tar = torch.randint(dialog_len, (5,))
            rand1_repl = torch.randint(len(self.cur_data), (5,))
            pair2 = copy.deepcopy(pos)
            
            #print(pair2)
            for i, j, k in zip(range(cluster_len), rand1_tar, rand1_repl):
                #pair2[i][j] = self.cur_data[k][i][torch.randint(len(self.cur_data[k][0]), (1,))] don't need that much possibility currently
                pair2[i][j] = self.cur_data[k][i][0]
            if dialog_len < 4:
                return pair2    
            
            #print(pair2)
            # disturb 2 turns
            rand2_repl = torch.randint(len(self.cur_data), (5,2))
            pair3 = copy.deepcopy(pos)
            
            for i,j in zip(range(cluster_len), rand2_repl):
                rand2_tar0 = torch.randint(2,(1,))
                if dialog_len % 2:
                    binary = math.floor(dialog_len/2) + int(rand2_tar0)
                else:
                    binary = math.floor(dialog_len/2)

                rand2_tar1 = torch.randint(binary,(2,))
                while rand2_tar1[0] == rand2_tar1[1]:
                    rand2_tar1 = torch.randint(binary,(2,))
                pair3[i][2*rand2_tar1[0]+(1-int(rand2_tar0))] = self.cur_data[j[0]][i][0]
                pair3[i][2*rand2_tar1[1]+(1-int(rand2_tar0))] = self.cur_data[j[1]][i][0]
            #print(pair3)
            return pair2, pair3

    @staticmethod
    def create_ref(pair, mode_me, i, j):
        ref = []
        dig_len = len(pair[0])
        if j in range(dig_len)[1:-1]:
            ref.append([pair[i][j-1]])
            ref.append([pair[i][j+1]])
        elif j == dig_len-2 and mode_me:
            ref = [[pair[k][-1]] for k in range(5)]
            ref.append([pair[i][j-1]])
        elif j == 0:
            ref.append([pair[i][j+1]])
        elif j == dig_len-1:
            if mode_me:
                ref = [[pair[k][-1]] for k in range(5) if k != dig_len-1]
            ref.append([pair[i][j-1]])
        ref.append([pair[i][j]])

        return ref

    @staticmethod
    def get_idx(ref, tar, used_idx=[], used_sen=[]):
        smooth = SmoothingFunction()
        bs = [sentence_bleu(ref, k) for k in tar]
        bs_pi = [{'bleu_score':i, 'idx':n} for n,i in enumerate(bs)]
        sorted_bs = sorted(bs_pi, key=lambda x:x['bleu_score'], reverse=True)
        #print(bs)
        #print(sorted_bs)
        k = 0
        best_idx = sorted_bs[k]['idx']
        if used_idx:
            while best_idx in used_idx:
                k += 1
                best_idx = sorted_bs[k]['idx']
        else:
            while tar[best_idx] in used_sen:
                k += 1    
                best_idx = sorted_bs[k]['idx']

        return best_idx
        


    @staticmethod
    def concat_with_sep(l_of_str):
        # plus sorting of two speaker
        pls_spkr = []
        for i in l_of_str:
            pls_spkr.append([])
            for idx, j in enumerate(i):
                if idx % 2 == 0:
                    pls_spkr[-1].append("A: "+j)
                else:
                    pls_spkr[-1].append("B: "+j)

        return ['</s></s>'.join(i) for i in pls_spkr]




if __name__ == '__main__':
    
    p = Path('dataset/dailydialog++')
    paths = list(p.iterdir())
    
    tn_data = Leveled_dataset('dataset/dailydialog++', split2four=False, level='medium',use_bleu=False)
    #print(len(tn_data.cur_data))
    
    tn_dataloader = DataLoader(tn_data, batch_size=1, shuffle=False)
    
    classify_data = []
    for i in tqdm(tn_dataloader):
        #classify_data = i
        print(i[0]['input_ids'][0][0][0].unsqueeze(0).size())
        print(i[1].size())
        #print(i[0])

        break
    
    #with open('classify_data.pkl', 'wb') as f:
    #   pkl.dump(classify_data, f)
    

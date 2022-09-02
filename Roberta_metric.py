import os
from typing import List

import torch
from torch import nn
#import texar.torch as tx
# from texar.torch.modules import BERTEncoder
from transformers import AutoTokenizer, AutoModel

#from util.config_base import Config


class RoBERTaMetric(nn.Module):

    NAME = 'Roberta_metric'

    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base')
        self.max_seq_length = args.max_seq_length

        self.backbone = AutoModel.from_pretrained(args.pretrained_model_name)

        roberta_hidden_size = self.backbone.config.hidden_size
        mlp_hidden_size_1 = int(roberta_hidden_size / 2)
        mlp_hidden_size_2 = int(mlp_hidden_size_1 / 2)
        '''
        self.mlp = nn.Sequential(
            nn.Linear(roberta_hidden_size, mlp_hidden_size_1),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_2, 1),
            nn.Sigmoid())
        '''
        self.turn_mlp = nn.Sequential(
            nn.Linear(roberta_hidden_size*2, roberta_hidden_size),
            nn.ELU()
        )

        self.diag_mlp = nn.Sequential(
            nn.Linear(roberta_hidden_size*2, mlp_hidden_size_1),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_2, 1),
            nn.Sigmoid()
        )

        self.device = torch.device("cuda".format(args.gpu))
        self.to(self.device)

        if hasattr(args, 'checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.checkpoint_dir_path, args.checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location='cuda:{}'.format(args.gpu))
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

        if hasattr(args, 'pretrain_checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.pretrain_checkpoint_dir_path,
                args.pretrain_checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location='cuda:{}'.format(args.gpu))
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

    def get_hidden_size(self):
        return self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask, indice=[]):
        output_dict = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True)
        pooled_output = output_dict['pooler_output']
        score = torch.tensor([0]) #self.mlp(pooled_output)
        
        #get dialog score
        last_hidden_state = output_dict['last_hidden_state'] #size:1*512*768
        left = 0
        last_hidden_state = last_hidden_state.squeeze(0)
        diag_rep = last_hidden_state[0].unsqueeze(0)
        for i in indice:
            right = i
            if right == 1: break

            if right == left+1:
                left=right
                continue
            
            turn_mean = torch.mean(last_hidden_state[int(left)+1:int(right)],0)
            turn_max,_ = torch.max(last_hidden_state[int(left)+1:int(right)],0)
            turn = torch.cat((turn_mean,turn_max),0)
            turn_rep = self.turn_mlp(turn)
            turn_rep = turn_rep.squeeze()
            #print(diag_rep.size())
            #print(turn_rep.size())
            diag_rep = torch.cat((diag_rep, turn_rep.unsqueeze(0)), 0)

            left = right

        diag1 = torch.mean(diag_rep, 0)
        diag2,_ = torch.max(diag_rep, 0)
        diag = torch.cat((diag1,diag2), 0)
        diag_score = self.diag_mlp(diag)

        return output_dict, score, diag_score


    @torch.no_grad()
    def get_score(self, sample: list):
        self.eval()
        input_ids, attention_mask, indice = self.encode_dialog(
            sample)
        _, score, dscore = self.forward(input_ids, attention_mask, indice)
        return score[0].item(), dscore[0].item()

    def encode_dialog(self, dialog: str):
        """Encodes the given context-response pair into ids.
        """
        
        tokenizer_outputs = self.tokenizer(
            dialog,
            return_tensors='pt', truncation=True,
            padding='max_length')
        input_ids = tokenizer_outputs['input_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        ts_sep = torch.ones(600)
        t_len = 0
        #print(input_ids.size())
        for n1,k in enumerate(input_ids[0]):
            if k == 2:
                ts_sep[t_len] = n1
                t_len += 1

        input_ids = input_ids.to(self.device)
        #token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        ts_sep = ts_sep.to(self.device)
        '''
        assert input_ids.size() == torch.Size([1, self.max_seq_length])
        assert token_type_ids.size() == torch.Size([1, self.max_seq_length])
        assert attention_mask.size() == torch.Size([1, self.max_seq_length])
        '''
        # tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print('context: ', context)
        # print('response: ', response)
        # print('tokenizer_outputs: ', tokenizer_outputs)
        # print('tokenized_text: ', ' '.join(tokenized_text))
        # print('length: ', len(tokenized_text))
        # exit()
        return input_ids,  attention_mask, ts_sep

    def encode_ctx_res_pair(self, context: List[str], response: str):
        """Encodes the given context-response pair into ids.
        """
        context = ' '.join(context)
        tokenizer_outputs = self.tokenizer(
            text=context, text_pair=response,
            return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_seq_length)
        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        assert input_ids.size() == torch.Size([1, self.max_seq_length])
        assert token_type_ids.size() == torch.Size([1, self.max_seq_length])
        assert attention_mask.size() == torch.Size([1, self.max_seq_length])
        # tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print('context: ', context)
        # print('response: ', response)
        # print('tokenizer_outputs: ', tokenizer_outputs)
        # print('tokenized_text: ', ' '.join(tokenized_text))
        # print('length: ', len(tokenized_text))
        # exit()
        return input_ids, token_type_ids, attention_mask

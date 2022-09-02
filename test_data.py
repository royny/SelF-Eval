import os
import json
import random

class HumanJudgementDatasetContest:

    DATA_MAP = {
        'dailydialog_EVAL': ['transformer_ranker', 'transformer_generator'],
        'empatheticdialogues': ['transformer_ranker', 'transformer_generator'],
        'convai2':['transformer_ranker', 'transformer_generator',
                   'bert_ranker', 'dialogGPT'],
    }




    def __init__(self,
                dataset_name, score_whole_dialog=True):
        self.dataset_name = dataset_name+'_eval.json'
        contest_data_dir = './human_evaluation_data'
        self.data_dir_path = os.path.join(contest_data_dir, self.dataset_name)
        if dataset_name in ['persona-see', 'fed-dial', 'dstc9']:
            #self.data_list = self._load_multi_turn_data()
            if score_whole_dialog:
                self.data_list = self._raw_multi_turn_data()
            else:
                self.data_list = self._load_multi_turn_data()
        else:
            self.data_list = self._load_data()



    def _raw_multi_turn_data(self):
        '''
            Score a whole dialog directly, the difference with load_multi_turn_data is it basically split a dialog with turns in a list.
        '''
        with open(self.data_dir_path, 'r') as f:
            data = json.load(f)

        data_list = []
        for num, i in enumerate(data):
            single_dialog = {}

            single_dialog['dialog'] = []


            for j in range(len(i['dialog'])):
                if j%2 == 0:
                    single_dialog['dialog'].append('A: '+i['dialog'][j]['text'])
                else:
                    single_dialog['dialog'].append('B: '+i['dialog'][j]['text'])

            human_score = {}
            for quality_name, score_list in i["annotations"].items():
                if len(score_list) != 0: 
                    score = sum(score_list)/len(score_list)
                    human_score[quality_name] = round(score, 2)
                else: human_score[quality_name] = 'NaN'
                #human_score[quality_name] = round(score, 2)
            single_dialog['human_score'] = human_score

            data_list.append(single_dialog)

        return data_list


    def _load_multi_turn_data(self):
        '''
            reshape multi turn data to several single turn data to score the whole dialog
            also list of dictionaries, in the 'turns' key, are results of spliting.
            {'turns':[{'context': , 'hyp_response': }, ... etc], 'human_score': {'quality1': s1, 'quality2': s2, etc}}
        '''
        with open(self.data_dir_path, 'r') as f:
            data = json.load(f)

        data_list = []
        for num, i in enumerate(data):
            single_dialog = {}
            
            single_dialog['turns'] = []
            human_score = {}
            if i['dialog'][0]['speaker'] == 'model': 
                for j in range(2, len(i['dialog']), 2):
                    turn = {}
                    turn['context'] = [i['dialog'][k]['text'] for k in range(j)]
                    turn['hyp_response'] = i['dialog'][j]['text']
                    single_dialog['turns'].append(turn)

            else:
                for j in range(0, len(i['dialog']), 2):
                    turn = {}
                    turn['context'] = [i['dialog'][k]['text'] for k in range(j+1)]
                    turn['hyp_response'] = i['dialog'][j+1]['text']
                    single_dialog['turns'].append(turn)
            
            
            human_score = {}
            
            for quality_name, score_list in i["annotations"].items():
                if len(score_list) != 0: 
                    score = sum(score_list)/len(score_list)
                    human_score[quality_name] = round(score, 2)
                else: human_score[quality_name] = 'NaN'
                #human_score[quality_name] = round(score, 2)
            single_dialog['human_score'] = human_score

            data_list.append(single_dialog)

        return data_list

    def _load_data(self):
        '''
            single turn data structure is the list of dictionaries
            {'context': context of the turn, 'hyp_response': model's response. 'human_score':{'quality1': s1, 'quality2': s2, etc}}
            when there is more than one quality, or
            {'context': context of the turn, 'hyp_response': model's response. 'human_score': s}
        '''
        with open(self.data_dir_path, 'r') as f:
            data = json.load(f)

        
        data_list = []

        if self.dataset_name == 'dstc10-task5.1_eval.json':

            for i in data:
                single_turn = {}
                single_turn['context'] = i['context'].strip().split('\n')
                single_turn['hyp_response'] = i['response']
                
                single_turn['human_score'] = round(random.random(), 2)
                #single_turn['quality_name'] = qualities
                data_list.append(single_turn)
            return data_list


        if self.dataset_name == 'fed-turn_eval.json':
            for i in data:
                cont = i['context'].strip().split('\n')
                raw_str = []
                for k in cont:
                    without_speaker = k.split(' ')
                    raw_str.append(' '.join(without_speaker[1:]))
                i['context'] = '\n'.join(raw_str)
                '''
                res = i['response'].strip().split(' ')
                i['response'] = ' '.join(res[1:])
                '''
        for i in data:
            single_turn = {}
            cont = i['context'].strip().split('\n')
            single_turn['context'] = []
            for k in range(len(cont)):
                if k%2 == 0:
                    single_turn['context'].append('A: '+cont[k])
                else:
                    single_turn['context'].append('B: '+cont[k])
            if len(cont) % 2 == 0:
                single_turn['hyp_response'] = 'A: '+i['response']
            else:
                single_turn['hyp_response'] = 'B: '+i['response']

            human_score = {}
            
            for quality_name, score_list in i['annotations'].items():
                #qualities.append(quality_name)
                score = sum(score_list)/len(score_list)
                human_score[quality_name] = round(score, 2)
            single_turn['human_score'] = list(human_score.items())[0][1] if len(human_score) == 1 else human_score
            #single_turn['quality_name'] = qualities
            data_list.append(single_turn)

        return data_list

    def __iter__(self):
        return self.data_list.__iter__()

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':

    dataset_name = 'fed-dial'
    eval_data = HumanJudgementDatasetContest(dataset_name)

    '''
            see where is the empty indices.
    '''
    multi_human_scores = {}
    empty_score_indices = []
    for i, sample in enumerate(eval_data):
        #if(i == 1): break
        #print(data['turns'])
        #print(data['hyp_response'])
        #print(data['human_score'])
        for qualitiy, score in sample['human_score'].items():
            if qualitiy not in multi_human_scores:
                multi_human_scores[qualitiy] = []
            if type(score)==float:  
                multi_human_scores[qualitiy].append(score)
            else:
                empty_score_indices.append(i)
    for i,j in multi_human_scores.items():
        print(i,':',len(j))
    print(empty_score_indices)

        

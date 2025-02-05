
import tqdm
import pickle
import logging as log
import torch
from torch.utils import data
import math

class Dataset(data.Dataset):
    def __init__(self, problem_number, concept_num, train_sample, root_dir, split='train'):
        super().__init__()
        self.map_dim = 0
        self.prob_encode_dim = 0
        self.path = root_dir
        self.problem_number = problem_number
        self.concept_num = concept_num
        self.show_len = 100
        self.split = split
        self.data_list = []
        self.train_sample = train_sample
        log.info('Processing data...')
        self.process()
        log.info('Processing data done!')
   

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def collate(self, batch):
        x, y = [], []
        seq_length = len(batch[0][-1][1]) 
        x_len = len(batch[0][-1][0][0])

        
        for i in range(0, seq_length):
            this_x = []
            for j in range(0, x_len):
                this_x.append([])
            x.append(this_x)
        item_num = len(batch[0])
        pre_x = [[] for i in range(0, item_num - 1)]
        for data in batch:
         
            this_seq_num, [this_x, this_y] = data[0], data[-1]
      
            for i in range(0, item_num - 1):
                pre_x[i].append(data[i])
            for i in range(0, seq_length):
                for j in range(0, x_len):
                    x[i][j].append(this_x[i][j])

            y += this_y[0 : this_seq_num]

        batch_x, batch_y =[], []
        for i in range(0, seq_length):
            x_info = []
            for j in range(0, x_len):
                if j != 5:
                    x_info.append(torch.tensor(x[i][j]))
                else:
                    x_info.append(torch.tensor(x[i][j]).float())
            batch_x.append(x_info)
        final_x = []
        for i in range(0, item_num - 1):
            final_x.append(torch.tensor(pre_x[i]))
        final_x.append(batch_x)
        return final_x, torch.tensor(y).float()
 
    def data_reader(self, stu_records):
        '''
        @params:
            stu_record: learning history of a user
        @returns:
            x: question_id, skills, interval_time_to_previous, concept_interval_time, elapsed_time, correctness 
            y: response
        '''
        x_list = []
        y_list = []
        concepts_interval_time_count = dict()
        '''interval time = 0, the interval time is much large'''

        for i in range(0, len(stu_records)):
            problem_id, skills, interval_time, elapsed_time, response= stu_records[i]
       
            operate = [1]
            if response == 0:
                operate = [0] 

            '''process the interval time'''
            for c_str in concepts_interval_time_count.keys():
                concepts_interval_time_count[c_str] += interval_time
            
            this_skill_str = ''
            for s in skills:
                this_skill_str += str(s) + '-'
            this_skill_str = this_skill_str[:-1]
            if not this_skill_str in concepts_interval_time_count.keys():
                concepts_interval_time_count[this_skill_str] = 0

            this_concept_interval = concepts_interval_time_count[this_skill_str]
            
            x_list.append([
                problem_id,
                skills,
                interval_time,
                this_concept_interval,
                elapsed_time,
                operate
            ])

            y_list.append(torch.tensor(response))

        return x_list, y_list

    def process(self):
        self.prob_encode_dim = int(math.log(self.problem_number,2)) + 1
        with open(self.path + 'history_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)
        loader_len = len(histories.keys())
        log.info('loader length: {:d}'.format(loader_len))
        proc_count = 0
        for k in tqdm.tqdm(histories.keys()):
            if len(self.data_list) >= self.train_sample and self.split == 'train':
                break
            stu_record = histories[k]
            if stu_record[0] < 10:
                continue
            dt = self.data_reader(stu_record[1])
            
            self.data_list.append([stu_record[0], dt])
            proc_count += 1
        log.info('final length {:d}'.format(len(self.data_list)))


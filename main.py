#!/usr/bin/env python
# encoding: utf-8
import torch
from torch.utils.data import DataLoader

import os
import pickle
import argparse
import logging as log

import models
from dataset import Dataset
import train



parser = argparse.ArgumentParser(description='Dsim')
parser.add_argument('--debug',          action='store_true',        help='log debug messages or not')
parser.add_argument('--run_exist',      action='store_true',        help='run dir exists ok or not')
parser.add_argument('--run_dir',        type=str,   default='run/result/junyi/batch_size64', help='dir to save log and models')
parser.add_argument('--data_dir',       type=str,   default='data/junyi500/') 
parser.add_argument('--log_every',      type=int,   default=0,      help='number of steps to log loss, do not log if 0')
parser.add_argument('--eval_every',     type=int,   default=10000,      help='number of steps to evaluate, only evaluate after each epoch if 0')
parser.add_argument('--save_every',     type=int,   default=50,      help='number of steps to save model')
parser.add_argument('--device',         type=int,   default=0,      help='gpu device id, cpu if -1')
parser.add_argument('--n_layer',type=int,   default=1,      help='number of mlp hidden layers in decoder')
parser.add_argument('--dim',type=int,   default=64,     help='hidden size for nodes')
parser.add_argument('--n_epochs',       type=int,   default=200,   help='number of epochs to train')
parser.add_argument('--batch_train',     type=int,   default=800,      help='number of instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-3,   help='learning rate')
parser.add_argument('--dropout',        type=float, default=0.0,   help='dropout') 
parser.add_argument('--seq_len',       type=int, default=200,   help='the length of the sequence') 
parser.add_argument('--gamma',        type=float, default=0.85,   help='graph_type') #
parser.add_argument('--plan',        type=str, default='', help='the training plan') 
parser.add_argument('--data_gen',          type=str,   default='ques_seq_gen',   help='run model')
parser.add_argument('--model',          type=str,   default='DSim',   help='run model')
parser.add_argument('--gmlp_layer',     type=int,   default=1,   help='the layer of the generator if implemented with MLP')

parser.add_argument('--multi_len',  type=int,   default=30,   help='the max length for testing continuous prediction') 
parser.add_argument('--batch_size',     type=int,   default=64,      help='number of instances in a batch')
parser.add_argument('--attention_heads',  type=int,   default=1,   help='the number of attention heads') 
parser.add_argument('--alpha_emb',  type=float,   default=0.1,   help='the value of alpha_emb') 
parser.add_argument('--alpha_bce',  type=float,   default=5.0,   help='the value of alpha_bce') 
parser.add_argument('--alpha_state',  type=float,   default=5.0,   help='the value of alpha_state') 
parser.add_argument('--alpha_diff',  type=float,   default=5.0,   help='the value of alpha_diff')
# predict num >= 1, 原来的2相当于现在的1
parser.add_argument('--predict_num',  type=int,   default=1,  help='the number augmented question, (predict_num + 1) % 4 == 0 is required')

parser.add_argument('--checkpoint_path',type=str,  default= 'run/edt/different_sample_num/train_500/plan3_dhkt_diff_33/diff_step_10_batch64_epoch400/params_100.pt',   help='the path of checkpoint')
parser.add_argument('--train_sample',  type=int,   default=6000,   help='the number of samples used for training')
parser.add_argument('--diff_time_step',  type=int,   default=10,   help='the time step for diffusion')


args = parser.parse_args() 

if args.debug:
    args.run_exist = True
    args.run_dir = 'debug'


dataset = ['a12', 'slp', 'ednet', 'juy']
for dtname in dataset:
    if dtname in args.data_dir:
        break

run_path = 'run/'+ dtname + '/different_sample_num/train_500/plan' + \
            args.plan + '_' + args.model + '/' + 'diff_step_' + str(args.diff_time_step) + '/'

if args.run_dir == 'None':
    args.run_dir  = run_path
os.makedirs(args.run_dir, exist_ok=args.run_exist)


log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.DEBUG if args.debug else log.INFO)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, \
                args.plan + '_' + args.model  + '_diff_step_' + str(args.diff_time_step)+ '_log.txt'), mode='w'))

log.info('args: %s' % str(args))
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

def preprocess():
    datasets = {}
    splits = ['train', 'valid', 'test']
    with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
        problem_number, concept_number, max_concept_of_problem = pickle.load(fp)
    setattr(args, 'max_concepts', max_concept_of_problem)
    setattr(args, 'concept_num', concept_number)
    setattr(args, 'problem_number', problem_number)

    for split in splits:
        file_name = os.path.join(args.data_dir, 'dataset_%s.pkl' % split)
        if 'cl' in args.model and 'train' in split:
            file_name = os.path.join(args.data_dir, 'dataset_%s_cl4kt.pkl' % split)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                datasets[split] = pickle.load(f)
            log.info('Dataset split %s loaded' % split)
        else:
            datasets[split] = Dataset(args.problem_number, args.concept_num, args.train_sample, root_dir=args.data_dir, split=split)
            with open(file_name, 'wb') as f:
                pickle.dump(datasets[split], f)
            log.info('Dataset split %s created and dumpped' % split)

    loaders = {}
    for split in splits:
        batch_size = args.batch_size if split == 'train' else args.batch_train
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            collate_fn=datasets[split].collate,
            shuffle=True if split == 'train' else False
        )

    return loaders

if __name__ == '__main__':
    
    loaders = preprocess()
    Model = getattr(models, args.model) 
    model = Model(args).to(args.device)
    log.info(str(vars(args)))
    train.train(model, loaders, args)

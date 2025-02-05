
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import modules
import numpy as np
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.distributions import multinomial

class DSim(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.ques_num = args.problem_number
        self.concept_num = args.concept_num
        self.max_concepts = args.max_concepts
        self.seq_length = args.seq_len
        self.time_step = args.diff_time_step
        self.alpha_emb = args.alpha_emb
        self.alpha_bce = args.alpha_bce
        self.alpha_state = args.alpha_state
        self.alpha_diff = args.alpha_diff
        self.max_concept = args.max_concepts
        self.tag_emb = nn.Parameter(torch.randn(2, args.dim).to(args.device), requires_grad=True)
        self.op_emb = nn.Parameter(torch.randn(2 * args.problem_number, args.dim).to(args.device), requires_grad=True)
        self.predict_num = args.predict_num
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim).to(args.device), requires_grad=True)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim).to(args.device), requires_grad=True)
        self.predictor = modules.funcs(args.n_layer, args.dim * 4, 1, args.dropout) 
        self.gru_h = modules.mygru(0, args.dim * 3, args.dim)
        showi0 = []
        for i in range(0, 10000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.device = args.device
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)
        self.gaussian_diff = modules.CondGaussianDiffusionX0N(timesteps=self.time_step, beta_schedule = 'cosine')
        self.unet = modules.mini_resnet(2, args.dim * 4, args.dim, 0.0)
        self.h_cat = modules.funcs(args.n_layer, self.node_dim * 2, self.node_dim, args.dropout)
        self.diff_state_est = modules.CondGaussianDiffusionX0N3(timesteps=self.time_step, \
                                                                beta_schedule = 'cosine',\
                                                                x_t_dim = 64)
        self.unet_state_est = modules.mini_resnet(2, args.dim * 4, args.dim, 0.0)
        
        self.attention_heads = args.attention_heads
        if args.attention_heads > 0:
            self.w_q_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_k_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_v_q = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 2, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_o_q = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)

            self.w_q_r = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_k_r = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_v_r = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_o_r = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)

            self.w_q_f = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_k_f = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_v_f = nn.ParameterList([nn.Parameter(torch.randn(args.dim * 1, args.dim).to(args.device), requires_grad=True) for i in range(0, self.attention_heads)])
            self.w_o_f = nn.Parameter(torch.randn( self.attention_heads, 1).to(args.device), requires_grad=True)
            # self.take_aug = self.predict_num + 1
        else:
            self.w_q_q = [torch.eye(args.dim * 2, args.dim)]
            self.w_k_q = [torch.eye(args.dim * 2, args.dim)]
            self.w_v_q = [torch.eye(args.dim * 2, args.dim)]
            self.w_o_q = torch.eye(1, 1)

            self.w_q_r = [torch.eye(args.dim, args.dim)]
            self.w_k_r = [torch.eye(args.dim, args.dim)]
            self.w_v_r = [torch.eye(args.dim, args.dim)]
            self.w_o_r = torch.eye(1, 1)

            self.w_q_f = [torch.eye(args.dim, args.dim)]
            self.w_k_f = [torch.eye(args.dim, args.dim)]
            self.w_v_f = [torch.eye(args.dim, args.dim)]
            self.w_o_f = torch.eye(1, 1)
            self.attention_heads = 1
            
        if self.predict_num == 1:
            self.integrate = modules.matrix_vec_light(args.n_layer, self.node_dim * 4, self.node_dim, args.dropout)
        else:
            self.integrate = modules.matrix_to_vec(args.dim, args.predict_num + 1,
                                               in_channels = 1, out_channels = 50, 
                                               vec_dim = args.dim * 4)
        self.norm1d = nn.LayerNorm(args.dim)
        self.norm2d = nn.LayerNorm(args.dim * 2)

        self.relation_matrix, \
            self.ques_concept_relation, \
            self.ques_freq = self.get_relation_matrix(args.data_dir)
        self.BCELoss = torch.nn.BCEWithLogitsLoss(reduce=True, reduction = 'mean')
        self.MSELoss = torch.nn.MSELoss(reduce=False, reduction = 'None')
    
    def get_relation_matrix(self, data_dir):
        with open(data_dir + 'problem_skills_relation.pkl', 'rb') as fp:
            relation = pickle.load(fp)
        relation_matrix_onehot = torch.zeros(self.ques_num - 1, self.concept_num - 1) + 1
        for k in relation.keys():
            tags = relation[k]
            for t in tags:
                if t != 0:
                    relation_matrix_onehot[k-1][t-1] = -1.0

        ques_concept_relation_list = [[0] * self.max_concept]
        for i in range(1, len(relation) + 1):
            this_append = relation[i]
            while(len(this_append) < self.max_concept):
                this_append.append(0)
            ques_concept_relation_list.append(this_append)
        
        with open(data_dir + 'ques_freq.pkl', 'rb') as fp:
            ques_freq = pickle.load(fp)
            

        return relation_matrix_onehot.to(self.device), \
            torch.tensor(ques_concept_relation_list).to(self.device), \
            torch.tensor(ques_freq[1:]).to(self.device)
    
    def aug_ques(self, prob_ids, related_concept_index):
        batch_size = prob_ids.size()[0]
        dist_probs = F.softmax(1 - self.ques_freq, dim = -1)
        distribute = multinomial.Multinomial(1, dist_probs)
        aug_ques = torch.argmax(distribute.sample((batch_size, self.predict_num)), dim = -1) + 1
        aug_concepts = self.ques_concept_relation[aug_ques]
        all_ques = torch.cat([prob_ids.unsqueeze(-1), aug_ques], dim = -1)
        all_concepts = torch.cat([related_concept_index.unsqueeze(1), aug_concepts], dim = 1)
        return all_ques, all_concepts
    
    def state_update(self, all_ques, denoised_results, ques_id, h):
        bs = all_ques.size()[0]
        tag = torch.tensor([1] + [0] * self.predict_num).to(self.device)
        tags_emb = self.tag_emb[tag].unsqueeze(0).repeat(bs, 1, 1)
        est_condition = torch.cat([all_ques, denoised_results], dim = -1)
        estimated_all_h = self.diff_state_est.p_sample_loop(self.unet_state_est, est_condition.unsqueeze(1)).squeeze(1)
        estimated_h = torch.mean(estimated_all_h, dim = 1)

        inte_condition = torch.cat([all_ques, denoised_results, tags_emb], dim = -1)
        inte_vec = self.integrate(inte_condition.unsqueeze(1))
        update_in = torch.cat([estimated_h, inte_vec, h], dim = -1)
        new_h = self.gru_h(update_in, h)
        return new_h 

    def get_diff_loss(self, ques_resp, operate, ques_id, h):
        bs = ques_resp.size()[0]
        op_emb = self.op_emb[operate.squeeze(1).long() + ques_id * 2 ].reshape(bs, 1, 1, self.node_dim)
        aug_op = torch.cat([op_emb, 
                              torch.zeros_like(op_emb).to(self.device).repeat(1, 1, self.predict_num, 1)], 
                              dim = -2)
        condition = torch.cat([h, ques_resp.squeeze(1)], dim = -1).reshape(bs, 1, 1, self.node_dim * 3)
        aug_condition = torch.cat([condition, torch.zeros_like(condition).repeat(1, 1, self.predict_num, 1)],
                                  dim = -2)
        
        # take_aug_op = aug_op.split([self.take_aug])
        # if self.take_aug == 1:
        #     aug_op = op_emb.split(1, self.predict_num, dim = 1)[0]
        #     aug_condition = aug_condition(1, self.predict_num, dim = 1)[0]

        split_op = aug_op.split([self.predict_num, 1], dim = -2)[0]
        split_condition = aug_condition.split([self.predict_num, 1], dim = -2)[0]

        loss = self.gaussian_diff.train_losses(self.unet, 
                                            split_op, 
                                            split_condition).reshape(bs, self.node_dim)

        # loss = self.gaussian_diff.train_losses(self.unet, 
        #                                     aug_op, 
        #                                     aug_condition).reshape(bs, self.node_dim)
        rand_prob = np.random.rand()
        if rand_prob < 0.5:
            return torch.mean(loss, dim = -1)
        else:
            return torch.mean(loss, dim = -1).detach()

    def prob_gaussian(self, h, all_ques_presentation):
        condition = torch.cat([h.unsqueeze(1).repeat(1, self.predict_num + 1, 1),
                               all_ques_presentation], dim = -1)
        denoised_result = self.gaussian_diff.p_sample_loop(
                                            self.unet, condition.unsqueeze(1)).squeeze(1)
        
        predict_in = torch.cat([denoised_result, 
                                h.unsqueeze(1).repeat(1, self.predict_num + 1, 1),
                                all_ques_presentation], dim = -1)
        all_probs = self.predictor(predict_in)
        prob = all_probs.squeeze(-1).split(
                                            [1, self.predict_num], dim = -1)[0]
        return prob, self.sigmoid(all_probs), denoised_result

    def cell(self, h, this_input):
        prob_ids, related_concept_index, _, _, _, operate = this_input
        batch_size = prob_ids.size()[0]
        all_ques, _ = self.aug_ques(prob_ids, related_concept_index)
        all_ques_presentation = self.get_ques(all_ques)
        
        prob, all_probs, denoised_result = self.prob_gaussian(h, all_ques_presentation)
        
        next_state = self.state_update( all_ques_presentation, denoised_result, all_ques, h)

        '''diffusion loss'''
        diff_loss = self.get_diff_loss(all_ques_presentation.split([1, self.predict_num], dim = 1)[0],
                                      operate, prob_ids, h)

        return next_state, prob, diff_loss, prob_ids, operate

    def get_ques(self, prob_ids):
        related_concept_index = self.ques_concept_relation[prob_ids]
        filter0 = torch.where(related_concept_index == 0, 
                              torch.tensor(0.0).to(self.device), 
                              torch.tensor(1.0).to(self.device))

        # data_len = prob_ids.size()[0]
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0)
        related_concepts = concepts_cat[related_concept_index]
        filter_sum = torch.sum(filter0, dim = -1)

        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            )
        
        concept_level_rep = torch.sum(related_concepts, dim = -2) / div.unsqueeze(-1)
     
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = -1)
        return v    
    
    def multi_head_attn(self, k, q, v, f_q, f_k, f_v, f_o):
        all_h = []
        for i in range(0, self.attention_heads):
            this_q = torch.matmul(q, f_q[i])#.unsqueeze(1)
            this_k = torch.matmul(k, f_k[i])
            this_v = torch.matmul(v, f_v[i])
            softmax = F.softmax( torch.matmul(this_q, this_k.transpose(2,1)) / math.sqrt(self.node_dim), dim = -1)
            this_h = torch.matmul(softmax, this_v)
            all_h.append(this_h.squeeze(1))
        atn_out = torch.matmul(
                torch.stack(all_h, dim = 3), f_o).squeeze(-1)    
        return atn_out

    def get_ground_states(self, ques_ids, operate):
        ques_resp = self.get_ques(ques_ids)
        operate_emb = self.op_emb[ques_ids * 2 + operate.long().squeeze(-1)]
        atn_out_en = self.multi_head_attn(self.norm2d(ques_resp), 
                                          self.norm2d(ques_resp),
                                          self.norm2d(ques_resp),
                                          self.w_q_q, self.w_k_q, self.w_v_q, self.w_o_q)
        o = atn_out_en

        atn_out_de = self.multi_head_attn(self.norm1d(operate_emb), 
                                          self.norm1d(operate_emb),
                                          self.norm1d(operate_emb),
                                          self.w_q_r, self.w_k_r, self.w_v_r, self.w_o_r)
        atn_out_de = operate_emb + atn_out_de

        atn_final_1 = self.multi_head_attn(self.norm1d(o), 
                                           self.norm1d(o), 
                                           self.norm1d(atn_out_de),
                                           self.w_q_f, self.w_k_f, self.w_v_f, self.w_o_f)
        atn_final_2 = atn_out_de + atn_final_1
        return atn_final_2

    def foward_function(self, inputs):
        probs, diff_loss, h_list = [], [], []
        ques_id_list, operate_list = [], []

        data_len = len(inputs[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        for i in tqdm(range(0, self.seq_length), leave=False): 
            h, prob, this_diff_loss, this_ques_id, this_operate = self.cell(h, inputs[1][i])
            probs.append(prob)
            h_list.append(h)
            ques_id_list.append(this_ques_id)
            operate_list.append(this_operate)
            diff_loss.append(this_diff_loss) 
        prob_tensor = torch.cat(probs, dim = 1)
        diff_loss_tensor = torch.stack(diff_loss, dim = 1)

        ground_states = self.get_ground_states(torch.stack(ques_id_list, dim = 1),
                                                torch.stack(operate_list, dim = 1))
        state_loss_tensor =  self.MSELoss(torch.stack(h_list, dim = 1), ground_states)
              
        return data_len, prob_tensor, state_loss_tensor, diff_loss_tensor
    
    # def gen_result(self, inputs):
    #     data_len, prob_tensor, _, _ = self.foward_function(inputs)
    #     return prob_tensor


    def get_result(self, inputs):
        data_len, prob_tensor, _, _ = self.foward_function(inputs)
        predict = []
        seq_num = inputs[0]
        for i in range(0, data_len):
            this_prob = prob_tensor[i][0 : seq_num[i]]
            predict.append(this_prob)
        return torch.cat(predict, dim = 0)

    def dhkt_hinge(self):
        p_emb_t = self.prob_emb
        c_emb_trans = self.concept_emb.transpose(1, 0)
        mat = torch.matmul(p_emb_t, c_emb_trans)
        loss = torch.mean(torch.clamp(1 + mat.mul(self.relation_matrix), min=0)) 
        return loss

    def get_loss(self, inputs, y):
        data_len, prob_tensor, state_loss_tensor, diff_loss_tensor = self.foward_function(inputs)
         
        diff_loss_list, predict, state_loss_list = [], [], []
        seq_num = inputs[0]
        for i in range(0, data_len):
            this_prob = prob_tensor[i][0 : seq_num[i]]
            predict.append(this_prob)

            this_state_loss = state_loss_tensor[i][0 : seq_num[i]]
            state_loss_list.append(this_state_loss)

            this_diff_loss = diff_loss_tensor[i][0: seq_num[i]]
            diff_loss_list.append(this_diff_loss)

        loss = self.BCELoss(torch.cat(predict, dim = 0), y) * self.alpha_bce + \
                self.dhkt_hinge() * self.alpha_emb + \
                torch.mean(torch.cat(state_loss_list, dim = 0)) * self.alpha_state + \
                torch.mean(torch.cat(diff_loss_list, dim = 0)) * self.alpha_diff
        return loss

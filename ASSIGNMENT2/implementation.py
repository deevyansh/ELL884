## This file will include the implementation of the crf model
## This assumes that the batch size comes first

import torch
from torch import nn
torch.manual_seed(1)
class CRF(nn.Module):
    def __init__(self, tags,start_tag, end_tag,pad_tag):
        super(CRF, self).__init__()
        self.transition=nn.Parameter(torch.zeros(tags,tags))
        self.start_tag=start_tag
        self.end_tag=end_tag
        self.pad_tag=pad_tag
        nn.init.uniform_(self.transition, -0.1, 0.1)
        self.transition.data[:, 1] = -1000.0
        self.transition.data[2,:] = -1000.0

    def forward(self,emissions,tags,mask):
        nll=-self.log_likelihood(emissions,tags,mask)
        return nll
    
    def log_likelihood(self,emission,tags,mask):
        score_single_tag=self.compute_score(emission,tags,mask)
        score_all_tags=self.compute_dp(emission,mask)
        return torch.sum(score_single_tag-score_all_tags)
    

    def compute_score(self,emission,tags,mask):
        bs,seq_length,tag_size=emission.shape
        scores=torch.zeros(bs)
        first_tags=tags[:,0] 
        transition_score=self.transition[self.start_tag,first_tags]
        emission_score = emission[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)
        
        scores+=transition_score+emission_score

        for i in range(1,seq_length):
            previous_tags=tags[:,i-1]
            current_tags=tags[:,i]
            transition_score=self.transition[previous_tags,current_tags]
            emission_score = emission[:, i].gather(1, current_tags.unsqueeze(1)).squeeze(1)

            transition_score=transition_score*mask[:,i]
            emission_score=emission_score*mask[:,i]
            scores+=transition_score+emission_score

        last_tags_indexes=mask.int().sum(1)-1
        last_tags=tags.gather(1,last_tags_indexes.unsqueeze(1)).squeeze(1)
        last_transition=self.transition[last_tags,self.end_tag]
        scores+=last_transition
        return scores
    
    def compute_dp(self,emission,mask):
        bs,seq_length,num_label=emission.shape
        alphas = torch.full((bs, num_label), 0)
        for i in range(seq_length):
            new_alphas=[]
            for label in range(num_label):
                emission_score=emission[:,i,label]
                emission_score=emission_score.unsqueeze(1)
                transition_score=self.transition[:,label]
                transition_score=transition_score.unsqueeze(0)
                temp=emission_score+transition_score+alphas
                new_alphas.append(torch.logsumexp(temp,dim=1))
            new_alphas = torch.stack(new_alphas, dim=1)
            new_mask=mask[:,i]
            new_mask=new_mask.unsqueeze(-1)
            alphas=(new_mask)*new_alphas+(1-new_mask)*alphas
        
        last_transition=self.transition[:,self.end_tag]
        last_transition=last_transition.unsqueeze(0)
        alphas=alphas+last_transition

        return torch.logsumexp(alphas,dim=1) 
    
    def viterbi_algorithm(self,emission,mask):
        bs,seq_length,num_label=emission.shape
        alphas = torch.full((bs, num_label), 0.0)

        backpointers=[]## dimension required(bs,seq_length,num_labels)
        for i in range(seq_length):
            new_alphas=[]
            temp_backpointers=[] ## dimension required(bs,num_labels)
            for label in range(num_label):
                emission_score=emission[:,i,label]
                emission_score=emission_score.unsqueeze(1)
                transition_score=self.transition[:,label]
                transition_score=transition_score.unsqueeze(0)
                temp=emission_score+transition_score+alphas
                max_score, max_score_tag = torch.max(temp, dim=-1)
                new_alphas.append(max_score)
                temp_backpointers.append(max_score_tag)
                

            temp_backpointers = torch.stack(temp_backpointers, dim=1)
            backpointers.append(temp_backpointers)
            new_alphas = torch.stack(new_alphas, dim=1)
            new_mask=mask[:,i]
            new_mask=new_mask.unsqueeze(-1)
            alphas=(new_mask)*new_alphas+(1-new_mask)*alphas
        backpointers = torch.stack(backpointers, dim=1)
        last_transition=self.transition[:,self.end_tag]
        last_transition=last_transition.unsqueeze(0)
        alphas=alphas+last_transition

        max_final_score,max_final_tag=torch.max(alphas,dim=1)


        new_ans = []
        temp = max_final_tag

        for i in range(seq_length):
            new_ans.append(temp)
            temp = backpointers[:, seq_length - i - 1].gather(1, temp.unsqueeze(1)).squeeze(1)

        # Reverse the order and stack tensors correctly
        new_ans.reverse()
        new_ans = torch.stack(new_ans, dim=1)
        new_ans = new_ans * mask
        return max_final_score, new_ans

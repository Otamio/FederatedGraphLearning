import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Config:
    def __init__(self):
        self.dim = 50

        
class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, device):
        C = Config()
        super(TransE, self).__init__()
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, C.dim).uniform_(-6 / math.sqrt(C.dim), 6 / math.sqrt(C.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(relation_num, C.dim).uniform_(-6 / math.sqrt(C.dim), 6 / math.sqrt(C.dim)), freeze=False)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        
        self.optimizer = torch.optim.SGD(self.parameters(), 1e-2, 0)
        
        self.dim = 50
        self.entity_num = entity_num
        self.relation_num = relation_num

        self.d_norm = 1
        self.gamma = torch.FloatTensor([1]).to(device)
    
    
    def normalize(self):
        entity_norm = torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / entity_norm
    
    
    def merge(self, model_A, model_B):
        self.entity_embedding.weight.data = (model_A.entity_embedding.weight.data + model_B.entity_embedding.weight.data) / 2
        self.relation_embedding.weight.data = (model_A.relation_embedding.weight.data + model_B.relation_embedding.weight.data) / 2

    
    def assign(self, model_A):
        self.entity_embedding.weight.data = model_A.entity_embedding.weight.data
        self.relation_embedding.weight.data = model_A.relation_embedding.weight.data
    
    
    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    
    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        pos_dis = self.entity_embedding(pos_head) + self.relation_embedding(pos_relation) - self.entity_embedding(
            pos_tail)
        neg_dis = self.entity_embedding(neg_head) + self.relation_embedding(neg_relation) - self.entity_embedding(
            neg_tail)
        return self.calculate_loss(pos_dis, neg_dis).requires_grad_()
    
    
    def calculate_loss(self, pos_dis, neg_dis):
        """
        :param pos_dis: [batch_size, embed_dim]
        :param neg_dis: [batch_size, embed_dim]
        :return: triples loss: [batch_size]
        """
        distance_diff = self.gamma + torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis, p=self.d_norm,
                                                                                            dim=1)
        return torch.sum(F.relu(distance_diff))

    
    def tail_predict(self, head, relation, tail, k=10):
        """
        to do tail prediction hits@k
        :param head: [batch_size]
        :param relation: [batch_size]
        :param tail: [batch_size]
        :param k: hits@k
        :return:
        """
        h_and_r = self.entity_embedding(head) + self.relation_embedding(relation)
        h_and_r = torch.unsqueeze(h_and_r, dim=1)
        h_and_r = h_and_r.expand(h_and_r.shape[0], self.entity_num, self.dim)
        embed_tail = self.entity_embedding.weight.data.expand(h_and_r.shape[0], self.entity_num, self.dim)
        values, indices = torch.topk(torch.norm(h_and_r - embed_tail, dim=2), k, dim=1, largest=False)
        tail = tail.view(-1, 1)
        return torch.sum(torch.eq(indices, tail)).item()
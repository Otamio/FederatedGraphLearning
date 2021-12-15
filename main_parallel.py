import pdb
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from load import Mapping
from model import TransE
from prepare_data import TrainSet, TestSet
import os
import torch.multiprocessing as mp

device = torch.device('cpu')
train_batch_size = 32
test_batch_size = 256
num_epochs = 50
top_k = 10


train_dataset = TrainSet()
test_dataset = TestSet()
test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                   test_dataset.raw_data)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


def train(model, epoch):
    
    total_loss = 0
    
    for batch_idx, (pos, neg) in enumerate(train_loader):

        pos, neg = pos.to(device), neg.to(device)
        pos = torch.transpose(pos, 0, 1)
        neg = torch.transpose(neg, 0, 1)

        loss = model(*pos, *neg)
        total_loss += loss.item()
        model.optimize(loss)
    
    
def main():    
    
    model = TransE(train_dataset.entity_num, train_dataset.relation_num, device)
    model.share_memory()
    
    for epoch in range(num_epochs):
        
        model.normalize()
        
        processes = []
        num_processes = 8
        
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model,epoch))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        corrct_test = 0
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            data = torch.transpose(data, 0, 1)
            corrct_test += model.tail_predict(data[0], data[1], data[2], k=top_k)
        print(f"===>epoch {epoch+1}, test accuracy {corrct_test/test_dataset.__len__()}")

    
if __name__ == "__main__":
    main()

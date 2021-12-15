import pdb
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from load import Mapping
from model import TransE
from prepare_data import TrainSet, TestSet


device = torch.device('cuda')
train_batch_size = 32
test_batch_size = 128
num_epochs = 50
top_k = 10


train_dataset = TrainSet()
train_dataset_A = TrainSet('./dataset/train1.txt')
train_dataset_B = TrainSet('./dataset/train2.txt')
test_dataset = TestSet()
test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                   test_dataset.raw_data)

train_loader_A = DataLoader(train_dataset_A, batch_size=train_batch_size, shuffle=True)
train_loader_B = DataLoader(train_dataset_B, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


def main():
        
    model = TransE(train_dataset.entity_num, train_dataset.relation_num, device)
    model_A = TransE(train_dataset.entity_num, train_dataset.relation_num, device)
    model_B = TransE(train_dataset.entity_num, train_dataset.relation_num, device)
    
    for epoch in range(num_epochs):
        
        model.normalize()
        model_A.normalize()
        model_B.normalize()
        model.cuda()
        model_A.cuda()
        model_B.cuda()
        total_loss = 0
        
        for batch_idx, (pos, neg) in enumerate(train_loader_A):
            pos, neg = pos.to(device), neg.to(device)
            pos = torch.transpose(pos, 0, 1)
            neg = torch.transpose(neg, 0, 1)
            loss = model_A(*pos, *neg)
            total_loss += loss.item()
            model_A.optimize(loss)
        
        for batch_idx, (pos, neg) in enumerate(train_loader_B):
            pos, neg = pos.to(device), neg.to(device)
            pos = torch.transpose(pos, 0, 1)
            neg = torch.transpose(neg, 0, 1)
            loss = model_B(*pos, *neg)
            total_loss += loss.item()
            model_B.optimize(loss)
            
        model.merge(model_A, model_B)
        model_A.assign(model)
        model_B.assign(model)
                            
        corrct_test = 0
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            data = torch.transpose(data, 0, 1)
            corrct_test += model.tail_predict(data[0], data[1], data[2], k=top_k)
        print(f"===>epoch {epoch+1}, test accuracy {corrct_test/test_dataset.__len__()}")

    
if __name__ == "__main__":
    main()


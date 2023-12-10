import torch
from torch.autograd import Variable
import time
import os
import sys

def train_epoch(epoch, data_loader, model, criterion, optimizer):
    model.train()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):

        inputs = inputs.cuda()
        targets = targets.cuda()
        
        inputs = Variable(inputs)
        targets = Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()

        if i % 10:
            print(f"Epoch: {epoch}, Loss: {loss}")
    
    return loss
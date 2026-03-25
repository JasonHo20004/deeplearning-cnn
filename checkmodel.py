# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
#from models.MDFNet_tuan_new_new_new import *
#from models.train_cifar10 import *
from models.BT1 import *

model = NetBT2()

model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))
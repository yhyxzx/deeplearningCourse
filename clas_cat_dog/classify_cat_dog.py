import os,shutil
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms,datasets
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from res_classification_net import ResNet

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
test_rate=0.2#训练集和测试集的比例为8:2。
img_num=12500
test_num=int(img_num*test_rate)


test_index = random.sample(range(0, img_num), test_num)
file_path= "kagglecatsanddogs_5340/PetImages"
tr="train"
te="test"
cat="Cat"
dog="Dog"

# #将上述index中的文件都移动到/test/Cat/和/test/Dog/下面去。
# for i in range(len(test_index)):
#     #移动猫
#     srcfile=os.path.join(file_path,tr,cat,str(test_index[i])+".jpg")
#     dstfile=os.path.join(file_path,te,cat,str(test_index[i])+".jpg")
#     mymovefile(srcfile,dstfile)
#     #移动狗
#     srcfile=os.path.join(file_path,tr,dog,str(test_index[i])+".jpg")
#     dstfile=os.path.join(file_path,te,dog,str(test_index[i])+".jpg")
#     mymovefile(srcfile,dstfile)




#定义transforms
transforms = transforms.Compose(
[

transforms.RandomResizedCrop(150),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

]

)

train_data = datasets.ImageFolder(os.path.join(file_path,tr), transforms)
test_data=datasets.ImageFolder(os.path.join(file_path,te), transforms)


batch_size=32
train_loader = data.DataLoader(train_data,batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader = data.DataLoader(test_data,batch_size=100)

# #架构会有很大的不同，因为28*28-》150*150,变化挺大的，这个步长应该快一点。
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1=nn.Conv2d(3,32,5,5)#和MNIST不一样的地方，channel要改成3，步长我这里加快了，不然层数太多。
#         self.conv2=nn.Conv2d(20,64,4,1)
#         self.fc1=nn.Linear(50*6*6,200)
#         self.fc2=nn.Linear(200,2)#这个也不一样，因为是2分类问题。
#     def forward(self,x):
#         #x是一个batch_size的数据
#         #x:3*150*150
#         x=F.relu(self.conv1(x))
#         #20*30*30
#         x=F.max_pool2d(x,2,2)
#         #20*15*15
#         x=F.relu(self.conv2(x))
#         #50*12*12
#         x=F.max_pool2d(x,2,2)
#         #50*6*6
#         x=x.view(-1,50*6*6)
#         #压扁成了行向量，(1,50*6*6)
#         x=F.relu(self.fc1(x))
#         #(1,200)
#         x=self.fc2(x)
#         #(1,2)
#         return x


lr=1e-3
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
model=ResNet().to(device)
optimizer=optim.Adam(model.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss()

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for idx,(t_data,t_target) in enumerate(train_loader):
        t_data,t_target=t_data.to(device),t_target.to(device)
        pred=model(t_data)#batch_size*2
        loss=loss_func(pred,t_target)

        #Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx%10==0:
            print("epoch:{},iteration:{},loss:{}".format(epoch,idx,loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (idx * 32) + (epoch * len(train_loader.dataset)))



def test(model,device,test_loader):
    model.eval()
    correct=0#预测对了几个。
    test_loss = 0
    with torch.no_grad():
        for idx,(t_data,t_target) in enumerate(test_loader):
            t_data,t_target=t_data.to(device),t_target.to(device)
            pred=model(t_data)#batch_size*2
            test_loss += loss_func(pred, t_target).item()
            pred_class = pred.data.max(1, keepdim=True)[1]
            correct+=pred_class.eq(t_target.view_as(pred_class)).sum().item()
    acc=correct/len(test_data)
    test_losses.append(test_loss/25)
    print("accuracy:{}".format(acc))


num_epochs=10
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(num_epochs+1)]
from time import *
begin_time=time()
test(model,device,test_loader)
for epoch in range(num_epochs):
    train(model,device,train_loader,optimizer,epoch)
    test(model,device,test_loader)
end_time=time()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

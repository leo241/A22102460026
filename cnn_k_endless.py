import numpy as np # cnn神经网络模型，理论上可以利用一直数值模拟的方法达到最优参数
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sample import * # 同目录下的sample.py
from tqdm import tqdm
import os


class MyDataset(Dataset): #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform


    def __getitem__(self, item): # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, label = self.data[item]
        img = self.transform(img)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([ # transform to figure, for further passing to nn
    transforms.ToTensor(), # ToTensor会给灰度图像自动增添一个维度
])

def create_folder(path):
    '''创建某路径文件夹'''
    if os.path.exists(path):
        return 0
    os.mkdir(path)


def get_data(train_num,valid_num,k):
    '''

    :param
    :return:
    '''
    train_list = list()
    valid_list = list()
    for i in tqdm(range(train_num)):
        line_real, object = random_n_sample(k)
        train_list.append([line_real, object[:, 0:2].reshape(1, -1)])  # X, y

    for i in tqdm(range(valid_num)):
        line_real, object = random_n_sample(k)
        valid_list.append([line_real, object[:, 0:2].reshape(1, -1)])  # X, y

    return train_list, valid_list

class cnn(nn.Module): # construction of netral network
    def __init__(self,n):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 # if stride = 1 padding = (kernel_size - 1)/2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        # 16 224 224
        self.conv2 = nn.Sequential( # 16,128,128
            nn.Conv2d(16,32,5,1,2), # 32 128 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 64 64
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),# 64 32 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 16 16
        )
        self.fc1 = nn.Linear(43008, 400)
        self.out= nn.Linear(400, n*2)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # print(x.size(), '进入全连接层前的维度')
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x

# hyper parameters
lr = 0.01
batch_size = 32 # how much data given to nn per iteration
EPOCH = 50 # you may decrease epoch, but maybe not too small
train_num = 32
valid_num = 1


if __name__ == '__main__':


    loss_func = nn.MSELoss()
    # training
    endless_epoch = 0
    while True: # endless sampling and training
        k = np.random.randint(1,11)
        endless_epoch += 1

        try:
            save_path = f'model_save{k}'
            create_folder(save_path)
            model_names = os.listdir(save_path)
            model_names_int = [int(model_name.replace('.pkl', '')) for model_name in model_names]
            first_iter = max(model_names_int)
            net = torch.load(save_path + '/' + str(first_iter) + '.pkl')
            print(f'\nmodel {first_iter} loaded successfully')

        except Exception as er:
            print(er)
            print('\nload weight fail! train from no weight...')
            net = cnn(k)
            first_iter = 0
        i = first_iter

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        train_list, valid_list = get_data(train_num, valid_num,k)  # 获取C0的训练集 验证集 和测试集
        train_data = MyDataset(train_list, transform=transform)
        valid_data = MyDataset(valid_list, transform=transform)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=0)  # batch_size是从这里的DataLoader传递进去的
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
        stop = False
        for epoch in tqdm(range(EPOCH)):
            if stop == True:
                break
            for step,(x,y) in enumerate(train_loader):
                output = net(x.float())
                # print(output, y.float())
                loss = loss_func(output, y.squeeze(1).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                if i % 10 == 0:
                    print(f'\niteration: {i}')
                    if i%200 == 0:
                        torch.save(net, f'{save_path}/{i}.pkl')
                        for data in valid_loader:
                            x_valid, y_valid = data
                            output = net(x_valid.float())
                            # print(output, y_valid.float())
                            valid_loss = loss_func(output, y_valid.squeeze(1).float())
                            print(f'\nendless:{endless_epoch} EPOCH:{epoch} iter:{i} num:{k}')
                            print('\ntrain_loss:', float(loss))
                            print('\n-----valid_loss-----:', float(valid_loss))
                            break

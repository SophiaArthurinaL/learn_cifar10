import torch
from torch import nn




class MyModel(nn.Module):
# 定义了一个名为 MyModel 的类，这个类继承自 nn.Module。
# 在 PyTorch 中，nn.Module 是所有神经网络模块的基类。所有自定义的模型类都应该继承自 nn.Module。


    def __init__(self, *args, **kwargs) -> None:
        # 卷积层和池化层
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # 全连接层
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)

        self.linear2 = nn.Linear(64,10)
        self.softmax = nn.Softmax(dim = 1)
    #前向传播
    #定义了前向传播过程，按顺序将输入数据通过各层，并最终输出预测结果。
    def forward(self,x):

        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.linear1(x)

        x = self.linear2(x)
        x = self.softmax(x)

        return x;


# x = torch.randn(1,3,32,32)
# myModel = MyModel()
# out = myModel(x)
# print(out)
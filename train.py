import torch.cuda
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from net import MyModel

# 初始化 TensorBoard
writer = SummaryWriter(log_dir ='logs')

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#加载数据集
train_data_set = datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
test_data_set = datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)

# 数据集大小
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

#加载数据集
train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=64, shuffle=False)

#定义网络
myModel = MyModel()

#用gpu训练
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU")
    myModel = myModel.cuda()

#训练轮数
epochs =300

# 定义损失函数和优化器
lossFn = torch.nn.CrossEntropyLoss()
optimizer = SGD(myModel.parameters(), lr=0.01)

for epoch in range(epochs):
    print("训练了 {} 轮，剩余 {} 轮".format(epoch + 1, epochs))

    # 损失变量
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 准确率
    train_total_acc = 0.0
    test_total_acc = 0.0

    # 开始训练
    myModel.train()
    for data in train_data_loader:
        inputs,labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()  # 清空梯度
        outputs = myModel(inputs)

        # 计算损失
        loss = lossFn(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 计算精度
        _, index = torch.max(outputs, 1)  # 得到预测值最大的值和下标
        acc = torch.sum(index == labels).item()

        train_total_loss += loss.item()
        train_total_acc += acc

    # 测试模型
    myModel.eval()
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = myModel(inputs)

            # 计算损失
            loss = lossFn(outputs, labels)

            # 计算精度
            _, index = torch.max(outputs, 1)  # 得到预测值最大的值和下标
            acc = torch.sum(index == labels).item()

            test_total_loss += loss.item()
            test_total_acc += acc

    # 输出训练和测试的损失与准确率
    print("train loss: {}, acc: {}, test loss: {}, acc: {}".format(
        train_total_loss, train_total_acc / train_data_size,
        test_total_loss, test_total_acc / test_data_size))

    # 记录到 TensorBoard
    writer.add_scalar('loss/train', train_total_loss, epoch + 1)
    writer.add_scalar('acc/train', train_total_acc / train_data_size, epoch + 1)
    writer.add_scalar('loss/test', test_total_loss, epoch + 1)
    writer.add_scalar('acc/test', test_total_acc / test_data_size, epoch + 1)

    # 每 50 个 epoch 保存一次模型
    if (epoch + 1) % 50 == 0:
        model_path = "model/model_{}.pth".format(epoch + 1)
        torch.save(myModel.state_dict(), model_path)
        print("Model saved at epoch {}".format(epoch + 1))

# 关闭 TensorBoard
writer.close()
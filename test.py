import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from net import MyModel

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 创建模型实例
model = MyModel()
use_gpu = torch.cuda.is_available() # 检查是否有GPU可用

if use_gpu:
    print("Using GPU")
    model = model.cuda() # 如果有GPU，将模型移动到GPU

# 加载状态字典
state_dict = torch.load('./model/model_300.pth', map_location=torch.device('cuda' if use_gpu else 'cpu'))
model.load_state_dict(state_dict) # 将状态字典加载到模型中

# 这一部分从文件model_300.pth加载模型的权重，并将这些权重加载到模型实例中。

# 定义图像变换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机水平翻转图像
    transforms.RandomCrop(32,padding=4),# 随机裁剪图像并添加填充
    transforms.ToTensor(),# 将图像转换为张量
    transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5)) # 标准化图像
])


#指定文件夹
folder_path = 'D:\\a004\\learn\\LearnPycharm\\testimages'
files = os.listdir(folder_path)  # 列出文件夹中的所有文件

# 得到每个文件的完整路径
images_files = [os.path.join(folder_path, f) for f in files]
print(images_files)


# 循环处理每张图像
for img in images_files:
    image = cv2.imread(img)  # 读取图像

    if image is None: # 检查图像是否成功加载
        print(f"Error: Could not load image at {folder_path}")
        continue

    cv2.imshow('image',image)  # 使用OpenCV显示图像
    image = cv2.resize(image, (32, 32)) # 调整图像大小为32x32像素
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 将图像从BGR转换为RGB
    image = Image.fromarray(image) # 将图像转换为PIL格式

    image = transform(image)  # 应用图像变换
    image = torch.reshape(image,(1,3,32,32)) # 重塑图像张量
    image = image.to('cuda' if use_gpu else 'cpu') # 将图像移动到适当的设备（GPU或CPU
    output = model(image) # 通过模型前向传播得到输出

    value, index = torch.max(output, 1) # 获取预测的类别和其概率
    index_value = index.item() # 获取预测的类别索引
    print(f"Predicted index: {index_value}")  # 打印预测的索引

    if index_value < len(classes): # 检查索引是否在类别范围内
        pre_val = classes[index_value]  # 获取预测的类别标签
        print("预测概率: {}, 预测下标: {}, 预测结果: {}".format(value.item(), index_value, pre_val))
    else:
        print(f"Index {index_value} is out of range for classes list.")

    cv2.waitKey() # 等待按键以关闭显示的图像
import torch
from matplotlib import pyplot as plt
# DataLoader类，用于批量加载数据
from torch.utils.data import DataLoader
# transforms是一个用来进行数据预处理和数据增强的模块
from torchvision import transforms
from torchvision import datasets

# 超参数设置
batch_size = 64
learning_rate = 0.01
# 冲量是一个优化方法，除了使用当前步的梯度外，还会加上之前步的动量向量
momentum = 0.5
EPOCH = 10

# 数据准备,将输入的PIL图像或numpy.ndarray转换为张量,(0.1307,)和(0.3081,)分别是MNIST数据集的均值和标准差。
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transforms,
                               download=True)  # 也可以先下载数据集，把download设置为false
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transforms, download=True)
# shuffle=True表示数据加载器在每个epoch开始时打乱数据的顺序，以提高模型训练的效果。如果shuffle=False，则数据按照原始顺序传递给模型。
# DataLoader将数据分成一批一批的样本
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 设计网络模型
# 设置网络模型的父类为nn.Module
class Net(torch.nn.Module):
    def __init__(self):
        # 使用super继承父类的属性和方法，torch.nn中有基本的卷积层，池化层，全连接层等组件
        super(Net, self).__init__()
        self.convl = torch.nn.Sequential(
            # 定义了一个二维卷积层，输入通道数为1（灰度图像），输出通道数为10，卷积核大小为5x5
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.fc = torch.nn.Sequential(
            # 比起torch.nn.Linear(320,10),多了一个隐藏层对输入进行特征提取和转换，提高模型的表达能力和泛化能力
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        # x是输入的张量，它的shape为 (batch_size, channels, height, width)。
        batch_size = x.size(0)
        x = self.convl(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()

# 构造损失函数和优化器
# 交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
# SGD是梯度下降算法，model.parameters()返回当前神经网络中的所有可训练参数，在使用优化器更新神经网络的过程中，需要使用这些可训练参数来计算梯度，momentum即使保持当前梯度下降变化方向的权重
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量


# 训练以及测试
# 把单独的一轮封装在函数里
def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    # 迭代训练集数据， 将训练集分成多个batch,从0个batch开始枚举所有的batch，batch_idx表示当前的batch索引，data表示当前的batch数据，包含了图片和标签。
    for batch_idx, data in enumerate(train_loader, 0):
        # 将data分成输入图片和标签两个类
        inputs, target = data
        # print(f"inputs.shape{inputs.shape}")
        # print(f"inputs{inputs}")
        # 清空上一次得到的梯度
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        # 根据损失值计算梯度，并进行反向传播
        loss.backward()
        # 使用优化器更新模型的参数，使得损失值尽可能小
        optimizer.step()

        # 把运算中loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        # torch.max第一个返回最大值，第二个返回最大值所在的位置，.data返回形状为[batch_size, num_classes]的张量，dim等于1表示在第一个维度进行取最大值的操作，得到每个样本的输出值的最大值，也就是每个样本的预测类别
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
        if batch_idx % 300 == 299:
            print('[%d, %5d]: loss: %.3f, acc: %.2f %%'  # %%%表示输出一个百分号
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_total = 0
            running_correct = 0
            running_loss = 0.0
    if epoch == 9:  # 保持最后的模型参数
        torch.save(model.state_dict(), './model_Mnist.pth')
        torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %%' % (epoch + 1, EPOCH, 100 * acc))
    return acc


if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test(epoch)
        acc_list_test.append(acc_test)
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()

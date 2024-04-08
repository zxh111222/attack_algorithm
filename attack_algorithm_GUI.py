import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import models


# 定义网络结构
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


# 加载预训练的MNIST分类模型
model = Net()
model.load_state_dict(torch.load("model_Mnist.pth", map_location=torch.device('cpu')))
model.eval()

image_path = "./mnist_images/"


# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    # 打印初始预测结果
    k_i = torch.argmax(model(image)).item()
    print("Initial Prediction:", k_i)

    # 找到梯度的符号
    sign_data_grad = data_grad.sign()
    # 将图像的每个像素沿梯度方向移动一小步
    perturbed_image = image + epsilon * sign_data_grad
    # 将图像的像素值裁剪到[0,1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# PGD攻击函数
def pgd_attack(image, epsilon, target, model, alpha, iterations):
    # 打印初始预测结果
    k_i = torch.argmax(model(image)).item()
    print("Initial Prediction:", k_i)

    perturbed_image = image.clone().detach()
    for i in range(iterations):
        # 将图像的梯度更新为攻击后的图像
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * torch.sign(data_grad)
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()
    return perturbed_image



def jsma_attack(image, target_label, model, theta=0.1, gamma=0.01, max_iter=100):
    # 打印初始预测结果
    k_i = torch.argmax(model(image)).item()
    print("Initial Prediction:", k_i)

    image.requires_grad = True
    pert_image = image.clone().detach()
    target_label = torch.tensor([target_label]).to(image.device)  # 修改此处为张量类型，并将目标标签移到设备上
    w = torch.zeros_like(image)

    for _ in range(max_iter):
        output = model(pert_image)
        loss = F.cross_entropy(output, target_label)
        model.zero_grad()
        loss.backward()

        # 检查梯度是否存在
        if pert_image.grad is None:
            break

        grad = pert_image.grad.data

        # 找到对输入像素的最大扰动
        max_grad_indices = torch.argmax(torch.abs(grad))

        # 对最大梯度像素进行扰动
        pert_image_flatten = pert_image.view(-1)
        pert_image_flatten[max_grad_indices] += gamma * torch.sign(grad.view(-1)[max_grad_indices])
        pert_image = pert_image_flatten.view_as(pert_image)

        # 将图像像素值限制在 [0, 1] 范围内
        pert_image = torch.clamp(pert_image, 0, 1)

        # 如果目标标签预测概率超过 theta，则停止攻击
        if output.argmax() == target_label.item() and output.max() > theta:
            break

    return pert_image






# DeepFool攻击函数
def deepfool_attack(image, net, max_iter=100, epsilon=1e-3):
    """
    :param image: 输入图像
    :param net: 目标模型
    :param max_iter: 最大迭代次数
    :param epsilon: 扰动大小
    :return: 对抗样本
    """
    image.requires_grad = True
    input_shape = image.size()
    pert_image = image.clone().detach()
    w = torch.zeros(input_shape).to(image.device)
    r_tot = torch.zeros(input_shape).to(image.device)
    k_i = torch.argmax(net(image)).item()  # 初始预测类别
    print("Initial Prediction:", k_i)
    itr = 0

    while k_i == torch.argmax(net(pert_image)).item() and itr < max_iter:
        # 计算梯度
        net.zero_grad()
        pert_image.requires_grad = True
        fs = net(pert_image)
        pert_image.grad = torch.autograd.grad(fs.max(), pert_image)[0]

        # DeepFool攻击步骤
        pert_labels = torch.argsort(fs)[0][-2:]
        pert_k_i = pert_labels[1].item()
        pert_f_k = fs[0, pert_k_i]
        pert_f_i = fs[0, k_i]
        pert_label_diff = pert_f_k - pert_f_i

        if pert_label_diff < epsilon:
            break

        w = (net(pert_image)[0, pert_k_i] - net(pert_image)[0, k_i]) * image.grad / torch.norm(image.grad, p=2)
        pert_image = pert_image + torch.clamp((pert_label_diff / torch.norm(w, p=2)) * w, -epsilon, epsilon)

        r_i = pert_image - image
        r_tot = r_tot + r_i

        itr += 1

    return image + r_tot




# 创建主窗口
root = tk.Tk()
root.title("Image Attack")
root.geometry("800x400")  # 设置窗口大小

# 创建左右两个 Frame
# left_frame = tk.Frame(root)
# left_frame.pack(side=tk.LEFT)
#
# right_frame = tk.Frame(root)
# right_frame.pack(side=tk.RIGHT)



# 加载图像函数
def load_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((200, 200))
    return ImageTk.PhotoImage(image)


# 定义全局变量
epsilon_fgsm = 0.8
epsilon_pgd = 0.8
alpha_pgd = 0.01
iterations_pgd = 40


# 攻击并显示函数
def attack_and_visualize(image_path):
    try:
        # 加载图像
        image = Image.open(image_path)
        # 转换图像为张量并添加批次维度
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        image_tensor.requires_grad = True

        # 将图像输入模型并获取输出
        output = model(image_tensor)
        _, target = torch.max(output, 1)  # 获取最大预测值的索引作为目标标签

        # 计算损失并进行反向传播
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # 获取图像的梯度
        data_grad = image_tensor.grad.data

        # # 展示原始图像
        # original_image_label.config(image=load_image(image_path))
        # original_image_label.update_idletasks()

        # 对图像进行攻击
        attack_algorithm = attack_algorithm_var.get()
        if attack_algorithm == 'FGSM':
            perturbed_image = fgsm_attack(image_tensor, epsilon_fgsm, data_grad)
        elif attack_algorithm == 'PGD':
            perturbed_image = pgd_attack(image_tensor, epsilon_pgd, target, model, alpha_pgd, iterations_pgd)
        elif attack_algorithm == 'DeepFool':
            perturbed_image = deepfool_attack(image_tensor, model)
        elif attack_algorithm == 'JSMA':
            target_label = 3  # 目标标签（例如：将数字5攻击为数字3）
            perturbed_image = jsma_attack(image_tensor, target_label, model)

        # 将攻击后的图像显示在GUI上
        perturbed_image_pil = transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())
        attacked_image_label.config(image=ImageTk.PhotoImage(perturbed_image_pil))
        attacked_image_label.update_idletasks()

        # 显示预测结果
        output_label.config(text="Predicted Label: {}".format(torch.argmax(output).item()))

    except Exception as e:
        print("Error:", e)
        # 可以添加适当的异常处理代码，例如弹出提示框提示用户

# 选择图像并加载显示
def select_image():
    filename = filedialog.askopenfilename(initialdir="./mnist_images", title="Select Image",
                                          filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if filename:
        # load_image(filename)
        attack_and_visualize(filename)
    # if filename:
    #     load_and_display_image(filename, left_frame)
    #     # 进行攻击并在右侧显示被攻击图像
    #     attack_and_visualize(filename)

# 加载图像并显示在标签中
def load_and_display_image(image_path, frame):
    image_label = tk.Label(frame)
    image_label.pack()
    image_label.config(image=load_image(image_path))



# 选择攻击算法的下拉菜单
attack_algorithm_label = tk.Label(root, text="Select Attack Algorithm:")
attack_algorithm_label.pack()

attack_algorithm_var = tk.StringVar()
attack_algorithm_combobox = ttk.Combobox(root, textvariable=attack_algorithm_var)
attack_algorithm_combobox['values'] = ('FGSM', 'PGD', 'DeepFool', 'JSMA')  # 可供选择的攻击算法
attack_algorithm_combobox.pack()

select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack()

# 原始图像展示区域
original_image_label = tk.Label(root)
original_image_label.pack()

# 被攻击后的图像展示区域
attacked_image_label = tk.Label(root)
attacked_image_label.pack()



# 创建显示预测结果的标签
output_label = tk.Label(root, text="Predicted Label: ")
output_label.pack()

root.mainloop()

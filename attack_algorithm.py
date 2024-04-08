import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt


class AdversarialAttack:
    def __init__(self, model_path="model_Mnist.pth"):
        # 初始化时加载模型
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else 'cpu')))
        self.model.eval()

    def jsma(self, image, ys_target, theta=1.0, gamma=0.1):
        """
        使用JSMA攻击方法对单个图像进行攻击

        Args:
        - image: 输入图像
        - ys_target: 目标标签
        - theta: 扰动步长
        - gamma: 扰动幅度

        Returns:
        - perturbed_image: 攻击后的图像
        - perturbed_label: 攻击后预测得到的标签
        """

        copy_sample = np.copy(image)
        var_sample = Variable(torch.from_numpy(copy_sample), requires_grad=True)

        outputs = self.model(var_sample)
        predicted = torch.max(outputs.data, 1)[1]
        # print('测试样本扰动前的预测值：{}'.format(predicted[0]))

        var_target = Variable(torch.LongTensor([ys_target, ]))

        if theta > 0:
            increasing = True
        else:
            increasing = False

        num_features = int(np.prod(copy_sample.shape[1:]))
        shape = var_sample.size()

        # 每次迭代扰动两个像素，因此max_iters被除以2.0
        max_iters = int(np.ceil(num_features * gamma / 2.0))

        # 掩码搜索域，如果像素已经达到顶部或底部，我们不再修改它。
        if increasing:
            search_domain = torch.lt(var_sample, 0.99)
        else:
            search_domain = torch.gt(var_sample, 0.01)
        search_domain = search_domain.view(num_features)


        output = self.model(var_sample)
        original_label = torch.max(output.data, 1)[1].cpu().numpy()

        iter = 0
        var_sample_flatten = None  # 初始化 var_sample_flatten
        while (iter < max_iters) and (original_label[0] != ys_target) and (search_domain.sum() != 0):
            # 计算前向导数的雅可比矩阵
            jacobian = self.compute_jacobian(self.model, var_sample)
            # 获取显著性地图并计算对分类最有影响的两个像素
            p1, p2 = self.saliency_map(jacobian, ys_target, increasing, search_domain, num_features)

            # 初始化 var_sample_flatten
            if var_sample_flatten is None:
                var_sample_flatten = var_sample.view(-1, num_features).clone()

            # 应用修改
            var_sample_flatten[0, p1] += theta
            var_sample_flatten[0, p2] += theta

            new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
            new_sample = new_sample.view(shape)
            search_domain[p1] = 0
            search_domain[p2] = 0
            var_sample = Variable(torch.tensor(new_sample), requires_grad=True)

            output = self.model(var_sample)
            original_label = torch.max(output.data, 1)[1].numpy()
            iter += 1

        perturbed_image = var_sample.data.cpu().numpy()
        perturbed_image_1 = var_sample.data
        perturbed_label = original_label

        return perturbed_image_1.squeeze(), perturbed_label[0]

    def compute_jacobian(self, model, input):
        """
        计算输入对应模型的雅可比矩阵

        Args:
        - model: PyTorch模型
        - input: 输入张量

        Returns:
        - jacobian: 雅可比矩阵
        """
        output = model(input)
        num_features = int(np.prod(input.shape[1:]))
        jacobian = torch.zeros([output.size()[1], num_features])
        mask = torch.zeros(output.size())  # 选择要计算的导数
        for i in range(output.size()[1]):
            mask[:, i] = 1
            input.grad = None  # 清零梯度
            output.backward(mask, retain_graph=True)
            # 将导数复制到目标位置
            jacobian[i] = input.grad.squeeze().view(-1, num_features).clone()
            mask[:, i] = 0  # 重置
        return jacobian

    def saliency_map(self, jacobian, target_index, increasing, search_space, nb_features):
        """
        计算显著性地图

        Args:
        - jacobian: 雅可比矩阵
        - target_index: 目标索引
        - increasing: 增加或减少
        - search_space: 搜索空间
        - nb_features: 特征数量

        Returns:
        - p: 最显著特征索引1
        - q: 最显著特征索引2
        """
        domain = torch.eq(search_space, 1).float()  # 搜索域
        # 所有特征对每个类的导数之和
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)
        target_grad = jacobian[target_index]  # 目标类的正向导数
        others_grad = all_sum - target_grad  # 其他类的正向导数之和

        # 将不在搜索域内的特征置零
        if increasing:
            increase_coef = 2 * (torch.eq(domain, 0)).float()
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0)).float()
        increase_coef = increase_coef.view(-1, nb_features)

        # 计算任意两个特征的目标类正向导数之和
        target_tmp = target_grad.clone().unsqueeze(0)  # 将target_tmp扩展为二维张量
        target_tmp -= increase_coef * torch.max(torch.abs(target_grad))
        alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
        # 计算任意两个特征的其他类正向导数之和
        others_tmp = others_grad.clone()
        others_tmp += increase_coef * torch.max(torch.abs(others_grad))
        beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

        # 将特征与自身相加的情况置零
        tmp = np.ones((nb_features, nb_features), int)
        np.fill_diagonal(tmp, 0)
        zero_diagonal = torch.from_numpy(tmp).byte()

        # 根据论文中显著性地图的定义（公式8和9），不满足要求的显著性地图中的元素将被置零。
        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)
        # 将掩码应用到显著性地图
        mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
        # 根据论文中的公式10进行乘法
        saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        # 获取最显著的两个像素
        max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
        p = max_idx // nb_features
        q = max_idx % nb_features
        return p, q


    def fgsm(self, index=100, epsilon=0.2):
        # 定义生成对抗样本的方法（FGSM算法）
        loss_function = nn.CrossEntropyLoss()
        image, label = testdata[index]
        image = image.unsqueeze(0)
        image.requires_grad = True
        outputs = self.model(image)
        loss = loss_function(outputs, torch.tensor([label]))
        loss.backward()

        # 计算梯度
        x_grad = torch.sign(image.grad.data)
        # 对抗样本生成
        x_adversarial = torch.clamp(image.data + epsilon * x_grad, 0, 1)

        # 预测对抗样本
        outputs = self.model(x_adversarial)
        predicted = torch.max(outputs.data, 1)[1]
        original_prediction = label
        attacked_prediction = predicted.item()

        return x_adversarial.squeeze(), original_prediction, attacked_prediction, label, predicted.item()

    def deepfool(self, img, label, max_iter=10, overshoot=0.02):
        orig_img = img.clone().detach()
        orig_img.requires_grad = True
        fs = self.model(orig_img)
        orig_label = torch.argmax(fs)

        pert_image = orig_img
        r_tot = torch.zeros_like(orig_img)

        loop_i = 0
        while torch.argmax(fs) == orig_label and loop_i < max_iter:
            pert = np.inf
            orig_img.grad = None  # 将梯度设为 None
            if fs[0, orig_label].requires_grad:  # 检查是否需要梯度
                fs[0, orig_label].backward(retain_graph=True)
                if orig_img.grad is None:  # 检查梯度是否为 None
                    break
                orig_grad = orig_img.grad.data.numpy().copy()

                for k in range(len(fs[0])):
                    if k == orig_label:
                        continue

                    orig_img.grad = None  # 将梯度设为 None
                    if fs[0, k].requires_grad:  # 检查是否需要梯度
                        fs[0, k].backward(retain_graph=True)
                        if orig_img.grad is None:  # 检查梯度是否为 None
                            break
                        cur_grad = orig_img.grad.data.numpy().copy()

                        w_k = cur_grad - orig_grad
                        f_k = (fs[0, k] - fs[0, orig_label]).data.numpy()

                        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = r_tot + torch.from_numpy(r_i).to(r_tot.device)

                with torch.no_grad():
                    pert_image = orig_img + (1 + overshoot) * r_tot
                    fs = self.model(pert_image)

            loop_i += 1

        return pert_image, orig_label, torch.argmax(fs), orig_img

    def pgd(self, image, label, epsilon=0.2, iter_eps=0.01, nb_iter=40, clip_min=0.0, clip_max=1.0, C=0.0, ord=np.inf,
            rand_init=True, flag_target=False):
        # PGD攻击算法
        loss_function = nn.CrossEntropyLoss()
        image = image.unsqueeze(0)



        x_tmp = image.clone().detach()
        perturbation = torch.zeros_like(image)

        for i in range(nb_iter):
            perturbation = self.single_step_attack(x_tmp, perturbation, label, epsilon, iter_eps, clip_min, clip_max, C,
                                                   ord, flag_target)
            perturbation = torch.Tensor(perturbation).type_as(image)

        adv_image = x_tmp + perturbation
        adv_image_1 = adv_image
        adv_image = adv_image.cpu().detach().numpy()

        adv_image = np.clip(adv_image, clip_min, clip_max)

        adv_image_gpu = torch.from_numpy(adv_image)
        outputs = self.model(adv_image_gpu)
        predicted = torch.max(outputs.data, 1)[1].item()

        return adv_image_1.squeeze(), label, predicted

    def single_step_attack(self, x, perturbation, label, epsilon, iter_eps, clip_min, clip_max, C, ord, flag_target):
        adv_x = x + perturbation
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True

        loss_function = nn.CrossEntropyLoss()
        preds = self.model(adv_x)

        if flag_target:
            loss = -loss_function(preds, torch.tensor([label]))
        else:
            loss = loss_function(preds, torch.tensor([label]))

        self.model.zero_grad()
        loss.backward()
        grad = adv_x.grad.data

        perturbation = iter_eps * torch.sign(grad)
        adv_x = adv_x.cpu().detach().numpy() + perturbation.cpu().numpy()
        x = x.cpu().detach().numpy()

        perturbation = np.clip(adv_x, clip_min, clip_max) - x
        perturbation = self.clip_perturbation(perturbation, ord, epsilon)

        return perturbation

    def clip_perturbation(self, perturbation, ord, epsilon):
        if ord == np.inf:
            perturbation = np.clip(perturbation, -epsilon, epsilon)
        else:
            raise NotImplementedError("Only L-infinity norm is supported.")

        return perturbation


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





if __name__ == "__main__":
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
    testdata = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)
    train_loader = DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

    attacker = AdversarialAttack()

    index = 100
    epsilon = 0.2

    image, label = testdata[index]


    # 使用 FGSM 攻击
    x_adv_fgsm, original_pred_fgsm, attacked_pred_fgsm, original_label_fgsm, attacked_label_fgsm = attacker.fgsm(
        index, epsilon)

    # 使用 DeepFool 攻击
    adversarial_image_deepfool, original_label_deepfool, attacked_label_deepfool, orginal_image = attacker.deepfool(image, label,
        overshoot=0.8, max_iter=10)

    # 使用 PGD 攻击
    adversarial_image_pgd, original_label_pgd, attacked_label_pgd = attacker.pgd(image, label)

    # 使用 JSMA 攻击
    adversarial_image_jsma, attacked_label_jsma = attacker.jsma(image, ys_target=2)

    plt.figure(figsize=(15, 12))  # 设置图形窗口大小

    img_adv_fgsm = transforms.ToPILImage()(x_adv_fgsm)
    plt.subplot(2, 3, 2)
    plt.title("FGSM Adversarial Image, label:{}".format(attacked_label_fgsm))
    plt.imshow(img_adv_fgsm)

    img_adv_deepfool = transforms.ToPILImage()(adversarial_image_deepfool)
    plt.subplot(2, 3, 3)
    plt.title("DeepFool Adversarial Image, label:{}".format(attacked_label_deepfool))
    plt.imshow(img_adv_deepfool)

    img_adv_pgd = transforms.ToPILImage()(adversarial_image_pgd)
    plt.subplot(2, 3, 4)
    plt.title("PGD Adversarial Image, label:{}".format(attacked_label_pgd))
    plt.imshow(img_adv_pgd)


    img_adv_jsma = transforms.ToPILImage()(adversarial_image_jsma)
    plt.subplot(2, 3, 5)
    plt.title("JSMA Adversarial Image, label:{}".format(attacked_label_jsma))
    plt.imshow(img_adv_jsma)


    img_org = transforms.ToPILImage()(testdata[index][0])
    plt.subplot(2, 3, 1)
    plt.title("Original Image, label:{}".format(label))
    plt.imshow(img_org)

    plt.show()


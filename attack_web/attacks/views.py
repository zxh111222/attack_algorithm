from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile
import torch
from torchvision import transforms
from .adversarial_attack import AdversarialAttack
import base64
import os
import csv
import uuid
import numpy as np
from PIL import Image
from io import BytesIO

# CSV文件路径
CSV_FILE_PATH = 'users.csv'


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 检查用户名是否已经存在
        if not is_user_exists(username):
            # 写入CSV文件
            with open(CSV_FILE_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([username, password])

            # 注册成功后重定向到登录页面
            return redirect('login')
        else:
            # 用户名已存在，返回注册页面并显示错误信息
            return render(request, 'attacks/register.html', {'error_message': '用户名已存在'})

    return render(request, 'attacks/register.html')


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # 检查用户名和密码是否匹配
        if is_valid_credentials(username, password):
            # 登录成功后设置session变量，并重定向到首页
            request.session['login_successful'] = True
            return redirect('index')
        else:
            # 登录失败，返回登录页面并显示错误信息
            return render(request, 'attacks/login.html', {'error_message': '用户名或密码错误'})

    return render(request, 'attacks/login.html')


def is_user_exists(username):
    # 检查用户名是否已经存在于CSV文件中
    with open(CSV_FILE_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username:
                return True
    return False


def is_valid_credentials(username, password):
    # 检查用户名和密码是否匹配
    with open(CSV_FILE_PATH, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == password:
                return True
    return False



def logout(request):
    # 清除登录状态
    if 'login_successful' in request.session:
        del request.session['login_successful']

    # 重定向到登录页面
    return redirect('login')



def index(request):
    return render(request, 'attacks/index.html')





def attack(request):
    error_message = None  # 用于存储错误消息
    if request.method == 'POST':
        # 获取攻击类型和上传的图像文件
        attack_type = request.POST.get('attack_type')
        image_file = request.FILES.get('image_file')
        ys_target = request.POST.get('ys_target')  # 获取 ys_target 参数

        if image_file:
            # 读取上传的图像文件
            original_image_pil = Image.open(image_file)

            # 将图像转换为 PyTorch 张量并调整大小（28x28）
            transform = transforms.Compose([transforms.Grayscale(),  # 转换为灰度图像
                                            transforms.Resize((28, 28)),  # 调整大小为 28x28
                                            transforms.ToTensor()])
            original_image_tensor = transform(original_image_pil).unsqueeze(0)

            # 使用 AdversarialAttack 类中的 predict_img 函数进行图像预测
            attack = AdversarialAttack()
            original_label = attack.predict_img(original_image_tensor)

            # 对图像执行攻击
            result = None  # 攻击结果的占位符

            # 执行所选择的攻击
            if attack_type == 'fgsm':
                result = attack.fgsm(original_image_tensor, original_label)
            elif attack_type == 'pgd':
                result = attack.pgd(original_image_tensor, original_label)
            elif attack_type == 'deepfool':
                result = attack.deepfool(original_image_tensor, original_label)
            elif attack_type == 'jsma':
                if ys_target:
                    ys_target_int = int(ys_target)
                else:
                    ys_target_int = 9
                result = attack.jsma(original_image_tensor, original_label, ys_target=ys_target_int)

            # 准备结果数据
            if result:
                adversarial_image_base64 = None
                if 'adversarial_image' in result:
                    # 将对抗图像转换为 base64，并调整大小（300x300）
                    adversarial_image_tensor = result['adversarial_image']
                    adversarial_image_pil = transforms.ToPILImage()(adversarial_image_tensor.cpu().squeeze())
                    adversarial_image_base64 = image_to_base64(adversarial_image_pil)

                # 将原始图像转换为 base64，并调整大小（300x300）
                # original_image_pil_resized = original_image_pil.resize((300, 300), Image.LANCZOS)
                # original_image_base64 = image_to_base64(original_image_pil_resized)
                # adversarial_image_pil = adversarial_image_pil.resize((300, 300), Image.LANCZOS)
                # adversarial_image_base64 = image_to_base64(adversarial_image_pil)

                original_image_pil = transforms.ToPILImage()(original_image_tensor.cpu().squeeze())
                original_image_base64 = image_to_base64(original_image_pil)


                result_data = {
                    'original_image': original_image_base64,
                    'adversarial_image': adversarial_image_base64,
                    'original_label': original_label,
                    'attacked_label': result['attacked_label']
                }

                return render(request, 'attacks/result.html', {'result': result_data})
            else:
                error_message = "攻击失败。"
        else:
            error_message = "未上传图像。"

    return render(request, 'attacks/index.html', {'error_message': error_message})




def image_to_base64(image):
    """
    将 PIL 图像对象转换为 base64 编码的字符串
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_encoded_string = base64.b64encode(img_byte_arr).decode('utf-8')
    return base64_encoded_string

def denormalize_image(image):
    # 将图像数据从 [0, 1] 转换为 [0, 255] 范围
    image = (image * 255).astype(np.uint8)
    return image



def save_adversarial_images(request):
    if request.method == 'POST':
        # 获取被攻击图像的 base64 编码数据列表
        adversarial_images_base64 = request.POST.getlist('adversarial_images_base64')

        # 设置保存图像的目录路径
        save_directory = './save_adversarial_images'  # 请替换为您希望保存图像的目录路径

        # 确保目录存在，如果不存在则创建
        os.makedirs(save_directory, exist_ok=True)

        # 保存图像
        saved_image_paths = []
        for idx, image_base64 in enumerate(adversarial_images_base64):
            # 将 base64 编码数据解码为图像数据
            image_data = base64.b64decode(image_base64)

            # 生成唯一的文件名
            filename = f'adversarial_image_{idx}_{uuid.uuid4().hex}.png'

            # 拼接保存图像的完整路径
            save_path = os.path.join(save_directory, filename)

            # 将图像数据写入到文件
            with open(save_path, 'wb') as f:
                f.write(image_data)

            # 添加保存的图像路径到列表中
            saved_image_paths.append(save_path)

        # 返回成功的 JSON 响应，包含保存的图像路径列表
        return JsonResponse({'message': 'Adversarial images saved successfully.', 'saved_image_paths': saved_image_paths})
    else:
        # 如果不是 POST 请求，则返回错误的 JSON 响应
        return JsonResponse({'error': 'Invalid request method.'})


def get_target_labels(request):
    # 从模型文件中获取标签信息，这里假设labels是一个包含所有标签的列表
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return JsonResponse(labels, safe=False)



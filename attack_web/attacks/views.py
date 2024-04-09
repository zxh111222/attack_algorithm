from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.base import ContentFile
from .adversarial_attack import AdversarialAttack
import base64
import os
import uuid
import numpy as np
from PIL import Image
from io import BytesIO

def index(request):
    return render(request, 'attacks/index.html')


def attack(request):
    if request.method == 'POST':
        # 获取请求中的参数
        attack_type = request.POST.get('attack_type')
        image_index = int(request.POST.get('image_index'))
        # 创建 AdversarialAttack 实例
        attack = AdversarialAttack()
        from PIL import Image
        # 先读取原始图片
        original_image_tensor, original_label = attack.read_img(image_index)
        # 将原始图片转换为 NumPy 数组
        original_image_np = original_image_tensor.numpy().squeeze()
        # 将 NumPy 数组反归一化
        original_image_np_denormalized = denormalize_image(original_image_np)
        # 将 NumPy 数组转换为 PIL 图像
        original_image_pil = Image.fromarray(np.uint8(original_image_np_denormalized))
        # 调整图像大小，保持灰度图像
        original_image_pil_resized = original_image_pil.resize((300, 300), Image.LANCZOS)
        # 将调整大小后的 PIL 图像转换为 base64 编码字符串
        original_image_base64 = image_to_base64(original_image_pil_resized)

        # 执行相应的攻击
        if attack_type == 'fgsm':
            result = attack.fgsm(image_index)
        elif attack_type == 'pgd':
            result = attack.pgd(image_index)
        elif attack_type == 'deepfool':
            result = attack.deepfool(image_index)
        elif attack_type == 'jsma':
            result = attack.jsma(image_index)
        else:
            result = None

        # 如果结果不为 None，处理图像数据
        if result:
            if 'adversarial_image' in result:
                # 将对抗图像转换为 NumPy 数组
                adversarial_image_np = result['adversarial_image'].numpy()
                adversarial_image_np_denormalized = denormalize_image(adversarial_image_np)
                # 将 NumPy 数组转换为 PIL 图像
                adversarial_image_pil = Image.fromarray(np.uint8(adversarial_image_np_denormalized))
                adversarial_image_pil = adversarial_image_pil.resize((300, 300))
                # 将 PIL 图像转换为 base64 编码字符串
                adversarial_image_base64 = image_to_base64(adversarial_image_pil)

            # 将处理后的结果添加到 result 字典中
            result = {
                'original_image': original_image_base64,
                'adversarial_image': adversarial_image_base64,
                'original_label': original_label,
                'attacked_label': result['attacked_label']
            }

        # 渲染结果页面
        return render(request, 'attacks/result.html', {'result': result})
    else:
        return HttpResponse("Invalid request")


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


from django.http import JsonResponse


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




import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap
from attack_algorithm import AdversarialAttack, Net
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.attacker = AdversarialAttack()  # 初始化攻击器

        # 创建主窗口
        self.setWindowTitle("Adversarial Attack Demo")
        self.setGeometry(100, 100, 800, 600)

        # 添加下拉框和标签
        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(10, 10, 150, 30)
        self.combo_box.addItem("FGSM")
        self.combo_box.addItem("PGD")
        self.combo_box.addItem("DeepFool")
        self.combo_box.addItem("JSMA")

        self.combo_box.activated.connect(self.attack_selected_algorithm)

        self.lbl_image = QLabel(self)
        self.lbl_image.setGeometry(200, 10, 400, 400)

        # 添加按钮来选择图像
        self.btn_choose_image = QPushButton("Choose Image", self)
        self.btn_choose_image.setGeometry(10, 50, 150, 30)
        self.btn_choose_image.clicked.connect(self.choose_image)

    def attack_selected_algorithm(self):
        algorithm = self.combo_box.currentText()
        if hasattr(self, 'image_path'):
            if algorithm == "FGSM":
                index = 0  # 指定图像索引
                epsilon = 0.2  # 设置 epsilon
                image, label = self.load_image(self.image_path)  # 加载图像
                adversarial_image, _, _, _, _ = self.attacker.fgsm(index, epsilon)  # FGSM 攻击
                self.show_image(adversarial_image)
            if algorithm == "PGD":
                index = 0  # 指定图像索引
                epsilon = 0.2  # 设置 epsilon
                image, label = self.load_image(self.image_path)  # 加载图像
                adversarial_image, _, _ = self.attacker.pgd(image, label, epsilon=epsilon)  # PGD 攻击
                self.show_image(adversarial_image)

            elif algorithm == "DeepFool":
                index = 0  # 指定图像索引
                image, label = self.load_image(self.image_path)  # 加载图像
                adversarial_image, _, _, _ = self.attacker.deepfool(image, label)  # DeepFool 攻击
                self.show_image(adversarial_image)

            elif algorithm == "JSMA":
                index = 0  # 指定图像索引
                image, label = self.load_image(self.image_path)  # 加载图像
                adversarial_image, _ = self.attacker.jsma(image, ys_target=2)  # JSMA 攻击
                self.show_image(adversarial_image)

    def load_image_function(self, image_path):
        """
        加载图像的方法

        Args:
        - image_path: 图像文件路径

        Returns:
        - image: 加载的图像
        """
        try:
            image = Image.open(image_path)
            return image
        except FileNotFoundError:
            print("File not found:", image_path)
            return None
        except Exception as e:
            print("Error loading image:", e)
            return None

    def get_label_from_image(self, image):
        """
        从图像中获取标签的方法

        Args:
        - image: PIL.Image对象

        Returns:
        - label: 图像的标签
        """
        # 这里假设图像文件名包含标签信息，例如 "image_label.jpg"
        # 您可能需要根据您的数据集和标签信息的存储方式进行适当调整
        file_name = image.filename  # 获取图像文件名
        label_str = file_name.split("_")[-1].split(".")[0]  # 提取标签字符串
        label = int(label_str)  # 将标签字符串转换为整数
        return label

    def load_image(self, image_path):
        # 加载图像并返回图像张量和标签
        image = self.load_image_function(image_path)  # 加载图像的方法，请根据实际情况替换为您自己的加载图像方法
        label = self.get_label_from_image(image)  # 从图像中获取标签的方法，请根据实际情况替换为您自己的方法
        return image, label

    def show_image(self, image_tensor):
        image_np = image_tensor.permute(1, 2, 0).numpy()
        plt.imshow(image_np)
        plt.show()

    def choose_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.png *.jpg)')
        pixmap = QPixmap(filename)
        self.lbl_image.setPixmap(pixmap)
        self.image_path = filename


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
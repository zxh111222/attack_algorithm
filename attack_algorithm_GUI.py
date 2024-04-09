import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QComboBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import torchvision
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
        self.setGeometry(100, 100, 1000, 800)

        # 添加下拉框和标签
        self.combo_box = QComboBox(self)
        self.combo_box.setGeometry(10, 10, 150, 30)
        self.combo_box.addItem("FGSM")
        self.combo_box.addItem("PGD")
        self.combo_box.addItem("DeepFool")
        self.combo_box.addItem("JSMA")

        self.combo_box.activated.connect(self.attack_selected_algorithm)

        self.lbl_image_orig = QLabel(self)
        self.lbl_image_orig.setGeometry(10, 50, 400, 400)

        self.lbl_image_adv = QLabel(self)
        self.lbl_image_adv.setGeometry(500, 50, 400, 400)

        self.lbl_label_orig = QLabel(self)
        self.lbl_label_orig.setGeometry(10, 470, 400, 30)

        self.lbl_label_adv = QLabel(self)
        self.lbl_label_adv.setGeometry(500, 470, 400, 30)

        # 添加按钮来选择图像
        self.btn_choose_image = QPushButton("Choose Image", self)
        self.btn_choose_image.setGeometry(10, 500, 150, 30)
        self.btn_choose_image.clicked.connect(self.choose_image)

        # 添加一个 widget 用于显示攻击后的图像
        self.adversarial_image_widget = QWidget(self)
        self.adversarial_image_widget.setGeometry(620, 10, 400, 400)
        self.adversarial_image_layout = QHBoxLayout(self.adversarial_image_widget)
        self.lbl_adversarial_image = QLabel(self.adversarial_image_widget)
        self.adversarial_image_layout.addWidget(self.lbl_adversarial_image)

        # 添加按钮来触发攻击
        self.btn_attack = QPushButton("Attack", self)
        self.btn_attack.setGeometry(10, 550, 150, 30)
        self.btn_attack.clicked.connect(self.attack_button_clicked)

        # 添加按钮来触发初始化操作
        self.btn_init = QPushButton("Init", self)
        self.btn_init.setGeometry(180, 550, 150, 30)
        self.btn_init.clicked.connect(self.init_button_clicked)
        self.btn_init.hide()  # 初始隐藏

        self.image_path = None

    def attack_button_clicked(self):
        self.attack_selected_algorithm()
        self.btn_attack.hide()
        self.btn_init.show()

    def init_button_clicked(self):
        self.lbl_image_orig.clear()
        self.lbl_image_adv.clear()
        self.lbl_label_orig.clear()
        self.lbl_label_adv.clear()
        self.image_path = None
        self.btn_init.hide()
        self.btn_attack.show()

    def attack_selected_algorithm(self):
        if self.image_path:
            algorithm = self.combo_box.currentText()
            image, label = self.load_image(self.image_path)  # 加载图像
            if algorithm == "FGSM":
                index = 0  # 指定图像索引
                epsilon = 0.2  # 设置 epsilon
                adversarial_image, _, _, _, attacked_label = self.attacker.fgsm(index, epsilon)  # FGSM 攻击
                adversarial_image = adversarial_image.unsqueeze(0)
                self.show_image(adversarial_image, attacked_label, self.lbl_adversarial_image)
            elif algorithm == "PGD":
                index = 0  # 指定图像索引
                epsilon = 0.2  # 设置 epsilon
                adversarial_image, _, attacked_label = self.attacker.pgd(image, label, epsilon=epsilon)  # PGD 攻击
                self.show_image(adversarial_image, attacked_label, self.lbl_adversarial_image)
            elif algorithm == "DeepFool":
                index = 0  # 指定图像索引
                adversarial_image, _, attacked_label, _ = self.attacker.deepfool(image, label)  # DeepFool 攻击
                self.show_image(adversarial_image, attacked_label, self.lbl_adversarial_image)
            elif algorithm == "JSMA":
                index = 0  # 指定图像索引
                adversarial_image, attacked_label = self.attacker.jsma(image, ys_target=2)  # JSMA 攻击
                self.show_image(adversarial_image, attacked_label, self.lbl_adversarial_image)
        else:
            print("Please choose an image first.")

    def load_image(self, image_path):
        # 加载图像并返回图像张量和标签
        image = self.load_image_function(image_path)  # 加载图像的方法，请根据实际情况替换为您自己的加载图像方法
        label = self.get_label_from_image(image)  # 从图像中获取标签的方法，请根据实际情况替换为您自己的方法
        return image, label

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


    def show_image(self, image_tensor, label, widget):
        # 将图像张量转换为可显示的图像
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # 设置图像部件
        pixmap = QPixmap.fromImage(QImage(image_np.data, image_np.shape[1], image_np.shape[0], QImage.Format_RGB888))
        widget.setPixmap(pixmap)
        widget.setScaledContents(True)

        # 设置标签部件
        widget.setText(f"Label: {label}")

    def show_images_labels(self, original_image, original_label, adversarial_image, adversarial_label):
        original_image_np = np.array(original_image)
        adversarial_image_np = np.array(adversarial_image)

        self.lbl_image_orig.setPixmap(QPixmap.fromImage(original_image))
        self.lbl_image_orig.setScaledContents(True)

        self.lbl_image_adv.setPixmap(QPixmap.fromImage(adversarial_image))
        self.lbl_image_adv.setScaledContents(True)

        self.lbl_label_orig.setText(f"Original Label: {original_label}")
        self.lbl_label_adv.setText(f"Adversarial Label: {adversarial_label}")

    def choose_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Images (*.png *.jpg)')
        pixmap = QPixmap(filename)
        self.lbl_image_orig.setPixmap(pixmap)
        self.lbl_image_orig.setScaledContents(True)
        self.image_path = filename



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

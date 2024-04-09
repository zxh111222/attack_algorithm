from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # 添加对根路径的处理
    path('attack/', views.attack, name='attack'),  # 添加对 attack/ 路径的处理
    path('save_adversarial_images/', views.save_adversarial_images, name='save_adversarial_images'),
]
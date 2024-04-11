from django.urls import path
from . import views

from django.urls import path
from . import views

urlpatterns = [
    path('', views.login, name='login'),  # 默认路径指向登录页面
    path('login/', views.login, name='login'),  # 登录页面路径
    path('register/', views.register, name='register'),  # 添加注册页面路径
    path('return_to_login/', views.login, name='return_to_login'),
    path('logout/', views.logout, name='logout'),
    path('index/', views.index, name='index'),  # 首页路径
    path('get_target_labels/', views.get_target_labels, name='get_target_labels'),
    path('attack/', views.attack, name='attack'),
    path('save_adversarial_images/', views.save_adversarial_images, name='save_adversarial_images'),
]

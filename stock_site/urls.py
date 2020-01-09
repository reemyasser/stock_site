"""stock_site URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from stock.views import p1,home,predition,form,insert,show,register,sign_up,sign_up_backend,log_in,log_in_backend,profile,log_out_backend,main_s

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',p1,name='index'),
    path('home/<str:username>',home,name="home"),
    path('prediction/',predition,name='prediction'),
    path('form/',form,name='form'),
    path( 'insert1/', insert,name='insert1'),
    path('show/',show,name='show'),
    path('register/',register,name='register'),
    path ('sign_up/',sign_up,name='sign_up'),
    path ('sign_up_backend/',sign_up_backend,name='sign_up_backend'),
    path('log/', log_in, name='log_in'),
    path('log_in_backend/', log_in_backend, name='log_in_backend'),
    path('profile/<str:username>/<str:stockdata>',main_s,name='profile'),
    path('log_out_backend/', log_out_backend, name='log_out_backend'),

]

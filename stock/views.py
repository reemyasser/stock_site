import io
import jinja2

import matplotlib.pyplot as plt, mpld3
import plotly
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
# Create your views here.
from . import models
from django.http import HttpResponse, HttpResponseRedirect
from .forms import staff_form
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from . import stock
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
import pandas as pd
from datetime import timedelta
def p1(request):
    return render(request,"index.html",{})
def home(request,username):
    return render(request,"home.html",{"u":username })
def predition(request):
    return render(request,"predition.html",{})
def form(request):
    return render(request,'form.html',{})
def insert(request):

    v1=request.POST['name']
    v2=request.POST['email']
    v3=request.POST['age']
    v4=request.POST['date']

    new1=models.company(fullname=v1,email=v2,ege=v3,birthday=v4)

    '''
    new.fullname=v1
    new.email = v2
    new.age = v3
    new.date = v4
    '''
    new1.save()

    return HttpResponse('home.html')

def show(request):
    data=models.company.objects.all()
    return render(request,'show.html',{'data':data})

#to send prarmater using where
def param(request,id):
    data=models.company.objeccts.all()

    return render(request, 'show.html', {'id':id,'data': data[id-1]})

def register (request):
    form=staff_form(request.POST or None)
    obj=models.company()
    if form.is_valid():
        obj.fullname=form.cleaned_data['name']
        obj.email=form.cleaned_data['email']
        obj.ege=form.cleaned_data['age']
        obj.birthday=form.cleaned_data['birthday']
        obj.save()
        return HttpResponseRedirect("/")
    return render(request,'register.html',{'f':form})

def sign_up(request):
    return render(request,'sign_up.html',{})
def sign_up_backend (request):
    user=User.objects.create_user(request.POST['username'],request.POST['email'],request.POST['password'])
    user.first_name=request.POST['first_name']
    user.last_name = request.POST['last_name']
    user.save()
    return HttpResponseRedirect('/')
def log_in(request):
    return render(request,'Log.html',{})
def log_in_backend(request):
    u=request.POST['username']
    p=request.POST['password']
    re=authenticate(username=u,password=p)
    if re is not None :
        print('log in')
        login(request,re)
        link='/home/'+str(re)
        return HttpResponseRedirect(link)
    else:
        return HttpResponse('user is not exsit ')
def profile (request,username,stockdata):
    return render(request, 'profile.html', {'u':username ,'stockdata':stockdata})
def log_out_backend(request):
    logout(request)
    return HttpResponseRedirect('/log')

def main_s (request,username,stockdata):
    s=request.GET.get('stockdata')

    data,dates,pred,list_three_days,testingy,acc_up=stock.main_stock(stockdata)
    data_html = data.to_html()
    testy=[]
    prediction=[]
    dates1=[]
    for i in reversed(testingy):
        testy.append(i)
    for i in reversed(pred):
        prediction.append(i)
    j=0
    for i in reversed(dates):
        dates[j]=pd.to_datetime(str(i)).strftime('%Y/%m/%d')
        dates1.append(i)
        j=j+1
    t = pd.to_datetime(str(dates1[0]))
    tomorowday = []
    for i in range(3):
        t = t + timedelta(days=1)
        tomorowday.append(t.strftime('%Y/%m/%d'))
        testy.append(list_three_days[i])
        dates.append(tomorowday[i])

    plt1,ax=stock.data_plotting(dates,prediction,testy,tomorowday,list_three_days)



    # everything after this is turning off stuff that's plotted by default


    g = mpld3.fig_to_html(plt1)
    print("ssssssssssss", stockdata)
    len1=len(dates)
    acc_down=100-float(acc_up)

    #graph_div = plotly.offline.plot(plt1, auto_open=False, output_type="div")


    list1=[{'list_three':z[0],'tomorowday':z[1]}for z in list( zip(list_three_days,tomorowday))[::-1]]

    list2=[{'dates':pd.to_datetime(str(z[0])).strftime('%Y/%m/%d'),'testingy':z[1][0],'pred':z[2][0]}for z in zip(dates1,testingy,pred)]
    return render(request,'profile.html',{'list1':list1,'list':list2,'tomorowday':tomorowday,'g':g,'acc_down':acc_down,'acc_up':acc_up,'stockdata':stockdata,'u':username,'data':data_html,'dates':dates1,'testingy':testingy,'pred':pred,'list_three':list_three_days})














def data_plotting(dates, pred, testingY):
    plt.figure('fig1')
    plt.plot(dates, pred, 'red')
    plt.plot(dates, testingY, 'blue')
    plt.xlabel('Date')
    plt.ylabel('open price')
    plt.show()

def pltToSvg():
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()
    return s

def get_svg(request):
    data_plotting() # create the plot
    svg = pltToSvg() # convert plot to SVG
    plt.cla() # clean up plt so it can be re-used
    response = HttpResponse(svg, content_type='image/svg+xml')
    return response
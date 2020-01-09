from django.db import models

# Create your models here.
class company (models.Model):
    fullname=models.CharField(max_length=100)
    email =models.EmailField()
    ege=models.IntegerField()
    birthday=models.DateField()





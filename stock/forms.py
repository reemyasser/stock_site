from django import forms
class staff_form(forms.Form):
    name=forms.CharField(required=True)
    email=forms.EmailField()
    age=forms.IntegerField()
    birthday=forms.DateField()
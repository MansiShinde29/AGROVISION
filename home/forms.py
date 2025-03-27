# home/forms.py

from django import forms
from .models import Contact
from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField()

class ContactUsForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = ['name', 'email', 'message']
        widgets = {
            'message': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Your message here'}),
        }

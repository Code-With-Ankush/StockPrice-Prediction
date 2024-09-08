from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm



#Creation user form
class UserForm(UserCreationForm):
    class Meta:
        model = User
        fields = {'username','password1','password2'}    




from django import forms 
from .models import Feedback

class FeedbackForm(forms.Form):
    SATISFACTION_CHOICES = [
        ('very_satisfied', 'Very Satisfied'),
        ('satisfied', 'Satisfied'),
        ('neutral', 'Neutral'),
        ('dissatisfied', 'Dissatisfied'),
        ('very_dissatisfied', 'Very Dissatisfied'),
    ]

    ACCURACY_CHOICES = [
        ('very_accurate', 'Very Accurate'),
        ('accurate', 'Accurate'),
        ('neutral', 'Neutral'),
        ('inaccurate', 'Inaccurate'),
        ('very_inaccurate', 'Very Inaccurate'),
    ]

    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Your Name'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your Email'}))
    satisfaction = forms.ChoiceField(choices=SATISFACTION_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))
    accuracy = forms.ChoiceField(choices=ACCURACY_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))
    improvements = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'What improvements would you suggest?'}), required=False)
    additional_feedback = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 5, 'placeholder': 'Any additional feedback?'}), required=False)


 # forms.py
from django import forms
from django import forms

class StockPredictionForm(forms.Form):
    # Basic stock input fields
    ticker = forms.CharField(label='Ticker Symbol', max_length=10)
    number_of_days = forms.IntegerField(label='Number of Days', min_value=1, max_value=365)

    # Choice of model
    model_type = forms.ChoiceField(
        label='Model Type',
        choices=[('LSTM', 'LSTM'), ('ARIMA', 'ARIMA')],
        initial='LSTM',
        widget=forms.RadioSelect
    )

    def clean(self):
        cleaned_data = super().clean()
        model_type = cleaned_data.get('model_type')
        
        # Validate model-specific parameters based on the chosen model type
        if model_type == 'ARIMA' and cleaned_data.get('seasonal'):
            if not cleaned_data.get('P') or not cleaned_data.get('D') or not cleaned_data.get('Q') or not cleaned_data.get('m'):
                raise forms.ValidationError(
                    "All seasonal parameters must be set for ARIMA when seasonal component is included."
                )

        return cleaned_data

        



from django import forms

class FundamentalDataForm(forms.Form):
    ticker = forms.CharField(label='Enter Stock Ticker', max_length=10)




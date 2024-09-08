from django.db import models

class Feedback(models.Model):
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

    name = models.CharField(max_length=100)
    email = models.EmailField()
    satisfaction = models.CharField(max_length=20, choices=SATISFACTION_CHOICES)
    accuracy = models.CharField(max_length=20, choices=ACCURACY_CHOICES)
    improvements = models.TextField()
    additional_feedback = models.TextField(blank=True)

    def __str__(self):
        return f"Feedback from {self.name}"


from django.db import models

class LSTMParameters(models.Model):
    lstm_units = models.IntegerField(default=50, help_text="Number of units in each LSTM layer")
    dropout_rate = models.FloatField(default=0.2, help_text="Dropout rate to prevent overfitting")
    epochs = models.IntegerField(default=25, help_text="Number of epochs for training the model")
    batch_size = models.IntegerField(default=32, help_text="Batch size for training")
    num_layers = models.IntegerField(default=2, help_text="Number of LSTM layers in the model")

    def __str__(self):
        return f"LSTM Params: Units={self.lstm_units}, Dropout={self.dropout_rate}, Epochs={self.epochs}, Batch Size={self.batch_size}, Layers={self.num_layers}"

class ARIMAParameters(models.Model):
    p = models.IntegerField(default=1, help_text="Order of the autoregressive part")
    d = models.IntegerField(default=1, help_text="Degree of first differencing involved")
    q = models.IntegerField(default=1, help_text="Order of the moving average part")
    seasonal = models.BooleanField(default=False, help_text="Include seasonal component?")
    P = models.IntegerField(default=0, help_text="Seasonal autoregressive order", blank=True, null=True)
    D = models.IntegerField(default=0, help_text="Seasonal differencing order", blank=True, null=True)
    Q = models.IntegerField(default=0, help_text="Seasonal moving average order", blank=True, null=True)
    m = models.IntegerField(default=0, help_text="Number of periods in each season", blank=True, null=True)

    def __str__(self):
        return f"ARIMA Parameters: p={self.p}, d={self.d}, q={self.q}, Seasonal={self.seasonal} (P={self.P}, D={self.D}, Q={self.Q}, m={self.m})"

import torch as nn
from mamba_ssm import Mamba

class NaiveForecast(nn.Module):
    def __init__(self):
        super(NaiveForecast, self).__init__()
        self.name = "Naive"
    
    def __call__(self, X):
        return X[:, -1, :]
    
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super(TimeSeriesLSTM, self).__init__()
        self.name = "LSTM"
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :]       # (batch_size, hidden_size)
        pred = self.fc(last_out)       # (batch_size, output_size)
        return pred
    
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super(TimeSeriesGRU, self).__init__()
        self.name = "GRU"
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)           # (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :]       # (batch_size, hidden_size)
        pred = self.fc(last_out)       # (batch_size, output_size)
        return pred

class TimeSeriesMamba(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(TimeSeriesMamba, self).__init__()
        self.name = "Mamba"

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.mamba = Mamba(d_model = hidden_size, d_state=8, d_conv=2, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)                
        out = self.mamba(x)
        x = self.dropout(x)  
        last_out = out[:, -1, :]
        pred = self.fc2(last_out)
        return pred
import torch.nn as nn

class AudioMaven(nn.Module):
    def __init__(self, num_classes, num_features, hidden_size=128, dropout_rate=0.3):
        super(AudioMaven, self).__init__()
        
        # 1D Convolutional layers to extract local patterns from MFCC features
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Global pooling for context
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # First fully connected layer after convolutional layers + ReLU
        self.fc1 = nn.Linear(128, hidden_size)
        self.relu = nn.ReLU()
        
        # LSTM layer (optional)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True,dropout=dropout_rate)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Final classification layer
        self.fc2 = nn.Linear(2*hidden_size, num_classes)
        

    def forward(self, x):

        # Conv1D + MaxPool
        x = nn.functional.relu(self.conv1(x))  # -> (batch, 64, feat_len/2)
        x = nn.functional.relu(self.conv2(x))  # -> (batch, 128, feat_len/4)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # x shape: (batch, 128, 1) -> (batch, 128)
        x = x.squeeze(-1)

        # Fully connected + ReLU
        x = self.relu(self.fc1(x))   # -> (batch, hidden_size)

        # LSTM expects (batch, seq_len, input_size)
        x = x.unsqueeze(1)           # -> (batch, 1, hidden_size)
        x, _ = self.lstm(x)          # -> (batch, 1, hidden_size*2)
        x = x.squeeze(1)             # -> (batch, hidden_size*2)

        # Dropout + Output
        x = self.dropout(x)
        x = self.fc2(x)              # -> (batch, num_classes)
        return x
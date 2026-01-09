import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        x = self.dropout(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        output = self.output_projection(x)
        
        return output

def train_transformer_full_state(train_input, train_target, val_input, val_target, epochtime=500, learning_rate=0.0005, batch_size=16, patience=20):
    model = TransformerModel(input_dim=36, d_model=36, num_heads=6, num_layers=2, d_ff=144, dropout=0.1)

    train_input_tensor = torch.FloatTensor(train_input)
    train_target_tensor = torch.FloatTensor(train_target)
    val_input_tensor = torch.FloatTensor(val_input)
    val_target_tensor = torch.FloatTensor(val_target)

    train_residual_tensor = train_target_tensor - train_input_tensor
    val_residual_tensor = val_target_tensor - val_input_tensor

    train_dataset = TensorDataset(train_input_tensor, train_residual_tensor)
    val_dataset = TensorDataset(val_input_tensor, val_residual_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochtime):
        model.train()
        total_train_loss = 0
        for batch_input, batch_residual in train_loader:
            optimizer.zero_grad()
            predicted_residual = model(batch_input)
            loss = criterion(predicted_residual, batch_residual)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_input, batch_residual in val_loader:
                predicted_residual = model(batch_input)
                val_loss = criterion(predicted_residual, batch_residual)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        train_rmse = float(np.sqrt(max(avg_train_loss, 0.0)))
        val_rmse = float(np.sqrt(max(avg_val_loss, 0.0)))
        print(f"Epoch {epoch+1}/{epochtime}, Train Loss: {avg_train_loss:.6f} (RMSE: {train_rmse:.6f}), Val Loss: {avg_val_loss:.6f} (RMSE: {val_rmse:.6f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history
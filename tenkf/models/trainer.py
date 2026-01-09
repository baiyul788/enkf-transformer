import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from config.experiment_config import ExperimentConfig


class TransformerTrainer:
    
    def __init__(self, config: ExperimentConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.lr
        )
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        print(f"Trainer initialized, device: {self.device}")
        print(f"Model parameters: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_data(self, train_input: np.ndarray, train_target: np.ndarray,
                    val_input: np.ndarray, val_target: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        train_residual = train_target
        val_residual = val_target
        
        train_input_tensor = torch.FloatTensor(train_input)
        train_residual_tensor = torch.FloatTensor(train_residual)
        val_input_tensor = torch.FloatTensor(val_input)
        val_residual_tensor = torch.FloatTensor(val_residual)
        
        train_dataset = TensorDataset(train_input_tensor, train_residual_tensor)
        val_dataset = TensorDataset(val_input_tensor, val_residual_tensor)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        print(f"Data preparation complete:")
        print(f"  Training samples: {len(train_input)}")
        print(f"  Validation samples: {len(val_input)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_rmse = 0.0
        num_batches = len(train_loader)
        
        for batch_input, batch_residual in train_loader:
            batch_input = batch_input.to(self.device)
            batch_residual = batch_residual.to(self.device)
            
            self.optimizer.zero_grad()
            predicted_residual = self.model(batch_input)
            loss = self.criterion(predicted_residual, batch_residual)
            
            loss.backward()
            self.optimizer.step()
            
            rmse = torch.sqrt(torch.mean((predicted_residual - batch_residual) ** 2))
            
            total_loss += loss.item()
            total_rmse += rmse.item()
        
        avg_loss = total_loss / num_batches
        avg_rmse = total_rmse / num_batches
        
        return avg_loss, avg_rmse
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_input, batch_residual in val_loader:
                batch_input = batch_input.to(self.device)
                batch_residual = batch_residual.to(self.device)
                
                predicted_residual = self.model(batch_input)
                loss = self.criterion(predicted_residual, batch_residual)
                
                rmse = torch.sqrt(torch.mean((predicted_residual - batch_residual) ** 2))
                
                total_loss += loss.item()
                total_rmse += rmse.item()
        
        avg_loss = total_loss / num_batches
        avg_rmse = total_rmse / num_batches
        
        return avg_loss, avg_rmse
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        print(f"\nStarting training, max epochs: {self.config.epochs}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            train_loss, train_rmse = self.train_epoch(train_loader)
            
            val_loss, val_rmse = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                improved = "✓"
            else:
                self.patience_counter += 1
                improved = " "
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} "
                  f"[{epoch_time:.1f}s] "
                  f"Train: {train_loss:.6f} (RMSE: {train_rmse:.6f}) "
                  f"Val: {val_loss:.6f} (RMSE: {val_rmse:.6f}) {improved}")
            
            if self.patience_counter >= self.config.patience:
                print(f"\nValidation loss did not improve for {self.config.patience} epochs, early stopping")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best validation RMSE: {min(self.history['val_rmse']):.6f}")
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def save_model(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {filepath}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            predicted_residual = self.model(input_tensor)
            
            if predicted_residual.dim() == 3:
                predicted_residual = predicted_residual.squeeze(0)
            
            return predicted_residual.cpu().numpy()
    
    def get_training_summary(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_rmse': min(self.history['val_rmse']) if self.history['val_rmse'] else None,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'total_epochs': len(self.history['train_loss']),
            'model_parameters': self._count_parameters(),
            'device': str(self.device)
        }
import numpy as np
import os
import json
import time
import glob
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'data_dir': 'dataset',
    'max_sims_per_type': 400,
    
    # Model architecture
    'features': [8, 16, 32, 64],
    'dropout': 0.05,
    
    # Input channels
    'use_T_t': True,
    'use_T_prev': True,
    'use_delta_T': True,
    
    # Training
    'batch_size': 16,
    'epochs': 60,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 6,
    
    # Data split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seed': 42,
    
    # Loss weights
    'loss_mse_weight': 0.6,
    'loss_l1_weight': 0.35,
    'loss_gradient_weight': 0.05,
    
    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    
    # Task
    'task_type': 'same_time',
}
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_experiment_dir():
    """Create timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    return exp_dir


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATASET LOADING - CORRECTED FOR FLAT STRUCTURE
# ============================================================================

def load_simulation_files(data_dir, bc_type=None, max_sims=None):
    """
    Load all simulation .npz files grouped by simulation.
    
    EXPECTED STRUCTURE:
        dataset/
        ├── dirichlet/
        │   ├── simulation_0001.npz
        │   ├── simulation_0002.npz
        │   └── ...
        └── neumann/
            ├── simulation_0001.npz
            ├── simulation_0002.npz
            └── ...
    
    Each .npz file contains:
        - T_norm: (N_frames, H, W)
        - liquid_frac: (N_frames, H, W)
        - times: (N_frames,)
        - bc_type, Q0, sigma, center_x, center_y, time_total
    """
    simulations = []
    
    bc_dirs = []
    if bc_type is None or bc_type == "dirichlet":
        bc_dirs.append(("dirichlet", os.path.join(data_dir, "dirichlet")))
    if bc_type is None or bc_type == "neumann":
        bc_dirs.append(("neumann", os.path.join(data_dir, "neumann")))
    
    for bc_name, bc_dir in bc_dirs:
        if not os.path.exists(bc_dir):
            print(f"Warning: {bc_dir} does not exist")
            continue
        
        # Find all .npz files directly in the directory (flat structure)
        npz_files = sorted(glob.glob(os.path.join(bc_dir, "*.npz")))
        
        if max_sims:
            npz_files = npz_files[:max_sims]
        
        for filepath in npz_files:
            try:
                data = np.load(filepath)
                
                # Extract filename without extension for identification
                sim_name = os.path.basename(filepath).replace('.npz', '')
                
                simulations.append({
                    'T_norm': data['T_norm'],           # (N_frames, H, W)
                    'liquid_frac': data['liquid_frac'], # (N_frames, H, W)
                    'times': data['times'],             # (N_frames,)
                    'metadata': {
                        'bc_type': str(data['bc_type']) if 'bc_type' in data else bc_name,
                        'sim_name': sim_name,
                        'Q0': float(data['Q0']) if 'Q0' in data else 0.0,
                        'sigma': float(data['sigma']) if 'sigma' in data else 0.0,
                        'center_x': float(data['center_x']) if 'center_x' in data else 0.0,
                        'center_y': float(data['center_y']) if 'center_y' in data else 0.0,
                        'time_total': float(data['time_total']) if 'time_total' in data else 0.0,
                    }
                })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue
    
    print(f"Loaded {len(simulations)} simulations")
    if len(simulations) == 0:
        print("  No data found! Please check dataset path.")
        print(f"  Looking in: {data_dir}/dirichlet/*.npz and {data_dir}/neumann/*.npz")
    
    return simulations


def split_simulations(simulations, train_ratio, val_ratio, test_ratio, seed):
    """Split by simulation (NO FRAME LEAKAGE)."""
    set_seed(seed)
    n_sims = len(simulations)
    indices = np.random.permutation(n_sims)
    
    n_train = int(n_sims * train_ratio)
    n_val = int(n_sims * val_ratio)
    n_test = n_sims - n_train - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_sims = [simulations[i] for i in train_indices]
    val_sims = [simulations[i] for i in val_indices]
    test_sims = [simulations[i] for i in test_indices]
    
    print(f"\nData split (by simulation - NO LEAKAGE):")
    print(f"  Train: {len(train_sims)} simulations")
    print(f"  Val:   {len(val_sims)} simulations")
    print(f"  Test:  {len(test_sims)} simulations")
    
    return train_sims, val_sims, test_sims


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class TemporalStefanDataset(Dataset):
    """
    Temporal dataset for Stefan problem.
    
    For each valid time step, creates:
        Input:  (T_t, T_{t-1}, ΔT)  (3 channels)
        Target: φ_t (liquid fraction)
    """
    def __init__(self, simulations, config):
        self.config = config
        self.samples = []
        
        for sim in simulations:
            T_seq = sim['T_norm']        # (N_frames, H, W)
            phi_seq = sim['liquid_frac'] # (N_frames, H, W)
            n_frames = T_seq.shape[0]
            
            # Need at least 2 frames for temporal info
            if n_frames < 2:
                continue
            
            # Determine frame range based on task
            if config['task_type'] == 'same_time':
                # Input: T_t, Output: φ_t (need T_{t-1} for delta)
                start_idx = 1 if (config['use_T_prev'] or config['use_delta_T']) else 0
                for t in range(start_idx, n_frames):
                    self._add_sample(T_seq, phi_seq, t, t, sim['metadata'])
            else:
                # 'forecast': Input: T_t, Output: φ_{t+1}
                for t in range(n_frames - 1):
                    self._add_sample(T_seq, phi_seq, t, t+1, sim['metadata'])
        
        print(f"  Created {len(self.samples)} temporal samples from {len(simulations)} simulations")
    
    def _add_sample(self, T_seq, phi_seq, input_idx, target_idx, metadata):
        """Create a single sample with multi-channel input."""
        H, W = T_seq.shape[1], T_seq.shape[2]
        channels = []
        
        # Current temperature (always included if enabled)
        if self.config['use_T_t']:
            channels.append(T_seq[input_idx])
        
        # Previous temperature
        if self.config['use_T_prev'] and input_idx > 0:
            channels.append(T_seq[input_idx - 1])
        
        # Temperature difference
        if self.config['use_delta_T'] and input_idx > 0:
            delta_T = T_seq[input_idx] - T_seq[input_idx - 1]
            channels.append(delta_T)
        
        # Fallback
        if len(channels) == 0:
            channels.append(T_seq[input_idx])
        
        X = np.stack(channels, axis=0).astype(np.float32)
        y = phi_seq[target_idx].astype(np.float32)
        
        self.samples.append({
            'X': X,
            'y': y,
            'metadata': metadata,
            'input_idx': input_idx,
            'target_idx': target_idx,
        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['X'], dtype=torch.float32),
            torch.tensor(sample['y'], dtype=torch.float32).unsqueeze(0)
        )


# ============================================================================
# LIGHTWEIGHT U-NET MODEL
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and optional dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout)
        )
    
    def forward(self, x):
        return self.mp_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution with sigmoid for [0,1] range."""
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv(x))


class TemporalUNet(nn.Module):
    """
    Lightweight Temporal U-Net for phase-field prediction.
    
    Input:  Multi-channel tensor (B, C, H, W)
    Output: Liquid fraction field (B, 1, H, W) in [0,1]
    """
    def __init__(self, in_channels, n_classes=1, features=[32, 64, 128, 256], dropout=0.1):
        super(TemporalUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout)
        
        # Decoder
        self.up1 = Up(features[3] * 2, features[3], dropout)
        self.up2 = Up(features[3], features[2], dropout)
        self.up3 = Up(features[2], features[1], dropout)
        self.up4 = Up(features[1], features[0], dropout)
        
        # Output
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combined regression loss for phase-field prediction.
    """
    def __init__(self, mse_weight=0.5, l1_weight=0.3, gradient_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def gradient_loss(self, pred, target):
        """Gradient difference loss to preserve interface sharpness."""
        # Sobel-like approximation
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        loss_x = self.mse(pred_grad_x, target_grad_x)
        loss_y = self.mse(pred_grad_y, target_grad_y)
        return (loss_x + loss_y) / 2
    
    def forward(self, pred, target):
        loss = 0.0
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(pred, target)
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1(pred, target)
        if self.gradient_weight > 0:
            loss += self.gradient_weight * self.gradient_loss(pred, target)
        return loss


# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator:
    """Calculate evaluation metrics for phase-field prediction."""
    
    @staticmethod
    def mse(y_true, y_pred):
        return float(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())))
    
    @staticmethod
    def r2(y_true, y_pred):
        return float(r2_score(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, threshold=0.5):
        """Dice coefficient for the melting front (φ >= 0.5)."""
        true_binary = (y_true >= threshold).astype(np.float32)
        pred_binary = (y_pred >= threshold).astype(np.float32)
        
        intersection = np.sum(true_binary * pred_binary)
        union = np.sum(true_binary) + np.sum(pred_binary)
        
        if union == 0:
            return 1.0
        return float(2 * intersection / union)
    
    @staticmethod
    def compute_all(y_true, y_pred):
        """Compute all metrics at once."""
        return {
            'MSE': MetricsCalculator.mse(y_true, y_pred),
            'MAE': MetricsCalculator.mae(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'R2': MetricsCalculator.r2(y_true, y_pred),
            'Dice': MetricsCalculator.dice_coefficient(y_true, y_pred),
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(y_pred.detach().cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = MetricsCalculator.compute_all(targets, preds)
    
    return total_loss / len(loader), metrics


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        total_loss += loss.item()
        all_preds.append(y_pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = MetricsCalculator.compute_all(targets, preds)
    
    return total_loss / len(loader), metrics


def train_model(model, train_loader, val_loader, config, exp_dir):
    """Full training pipeline with early stopping."""
    device = config['device']
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(
        mse_weight=config['loss_mse_weight'],
        l1_weight=config['loss_l1_weight'],
        gradient_weight=config['loss_gradient_weight']
    )
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(exp_dir, 'model_best.pt'))
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        # Progress logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"MAE: {train_metrics['MAE']:.4f}/{val_metrics['MAE']:.4f} | "
                  f"Dice: {train_metrics['Dice']:.4f}/{val_metrics['Dice']:.4f} | "
                  f"Time: {elapsed:.0f}s")
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'metrics': val_metrics
    }, os.path.join(exp_dir, 'model_final.pt'))
    
    # Save history
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        # Convert numpy values to Python types
        serializable_history = {}
        for key, values in history.items():
            if key in ['train_metrics', 'val_metrics']:
                serializable_history[key] = [{k: float(v) for k, v in m.items()} for m in values]
            else:
                serializable_history[key] = [float(v) for v in values]
        json.dump(serializable_history, f, indent=4)
    
    print(f"\nTraining completed in {time.time() - start_time:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model, history


# ============================================================================
# INFERENCE AND SAVING
# ============================================================================

def run_inference(model, test_loader, device, exp_dir):
    """Run inference on test set and save results."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_errors = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y_pred = model(X).cpu().numpy()
            y_true = y.numpy()
            
            all_preds.append(y_pred)
            all_targets.append(y_true)
            all_errors.append(np.abs(y_true - y_pred))
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    errors = np.concatenate(all_errors, axis=0)
    
    # Save as compressed NPZ
    np.savez_compressed(os.path.join(exp_dir, 'predictions.npz'), arr=preds)
    np.savez_compressed(os.path.join(exp_dir, 'ground_truth.npz'), arr=targets)
    np.savez_compressed(os.path.join(exp_dir, 'errors.npz'), arr=errors)
    
    # Compute and save metrics
    metrics = MetricsCalculator.compute_all(targets, preds)
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nTest Metrics:")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")
    print(f"  Dice: {metrics['Dice']:.6f}")
    
    return preds, targets, errors, metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, exp_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    train_mae = [m['MAE'] for m in history['train_metrics']]
    val_mae = [m['MAE'] for m in history['val_metrics']]
    axes[0, 1].plot(train_mae, 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(val_mae, 'r-', label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice coefficient
    train_dice = [m['Dice'] for m in history['train_metrics']]
    val_dice = [m['Dice'] for m in history['val_metrics']]
    axes[1, 0].plot(train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1, 0].plot(val_dice, 'r-', label='Val Dice', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].set_title('Interface Dice Coefficient')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_history.png'), dpi=150)
    plt.close()


def create_preview_frames(model, test_loader, device, exp_dir, n_samples=4):
    """Create preview frames showing model predictions."""
    model.eval()
    
    # Get a batch of test samples
    data_iter = iter(test_loader)
    X_batch, y_batch = next(data_iter)
    
    with torch.no_grad():
        X_batch = X_batch[:n_samples].to(device)
        y_pred = model(X_batch).cpu().numpy()
        y_true = y_batch[:n_samples].numpy()
        X_input = X_batch.cpu().numpy()
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Input (first channel only - T_t)
        axes[i, 0].imshow(X_input[i, 0], cmap='hot', origin='lower')
        axes[i, 0].set_title('Input: Current Temperature')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(y_true[i, 0], cmap='coolwarm', origin='lower', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(y_pred[i, 0], cmap='coolwarm', origin='lower', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction (MAE: {np.mean(np.abs(y_true[i,0] - y_pred[i,0])):.4f})')
        axes[i, 2].axis('off')
        
        # Error
        error = np.abs(y_true[i, 0] - y_pred[i, 0])
        im = axes[i, 3].imshow(error, cmap='Reds', origin='lower', vmin=0, vmax=0.5)
        axes[i, 3].set_title('Error Map')
        axes[i, 3].axis('off')
    
    plt.colorbar(im, ax=axes[:, 3], shrink=0.5)
    plt.suptitle('Temporal U-Net Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'predictions_preview.png'), dpi=150)
    plt.close()
    
    print(f"Preview saved to {exp_dir}/predictions_preview.png")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("PUREAI - Data-Driven Temporal U-Net for Stefan Problem")
    print("No physics-informed components. Pure data-driven surrogate model.")
    print("=" * 70)
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Create experiment directory
    exp_dir = create_experiment_dir()
    print(f"\nExperiment directory: {exp_dir}")
    
    # 1. Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Dataset")
    print("=" * 70)
    
    simulations = load_simulation_files(
        CONFIG['data_dir'],
        bc_type=None,
        max_sims=CONFIG['max_sims_per_type']
    )
    
    if len(simulations) == 0:
        print("\nError: No simulations found. Please check dataset path.")
        print(f"Expected structure:")
        print(f"  {CONFIG['data_dir']}/dirichlet/*.npz")
        print(f"  {CONFIG['data_dir']}/neumann/*.npz")
        return
    
    # 2. Split by simulation (NO FRAME LEAKAGE)
    print("\n" + "=" * 70)
    print("STEP 2: Splitting Data (By Simulation)")
    print("=" * 70)
    
    train_sims, val_sims, test_sims = split_simulations(
        simulations,
        CONFIG['train_ratio'],
        CONFIG['val_ratio'],
        CONFIG['test_ratio'],
        CONFIG['seed']
    )
    
    # 3. Create datasets
    print("\n" + "=" * 70)
    print("STEP 3: Creating Temporal Datasets")
    print("=" * 70)
    
    train_dataset = TemporalStefanDataset(train_sims, CONFIG)
    val_dataset = TemporalStefanDataset(val_sims, CONFIG)
    test_dataset = TemporalStefanDataset(test_sims, CONFIG)
    
    # Determine input channels
    sample_X, _ = train_dataset[0]
    n_channels = sample_X.shape[0]
    print(f"Input channels: {n_channels}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 4. Create model
    print("\n" + "=" * 70)
    print("STEP 4: Creating Model")
    print("=" * 70)
    
    model = TemporalUNet(
        in_channels=n_channels,
        n_classes=1,
        features=CONFIG['features'],
        dropout=CONFIG['dropout']
    )
    print(f"Model: Temporal U-Net (Lightweight)")
    print(f"Features: {CONFIG['features']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Train
    print("\n" + "=" * 70)
    print("STEP 5: Training")
    print("=" * 70)
    
    model, history = train_model(model, train_loader, val_loader, CONFIG, exp_dir)
    
    # 6. Plot training history
    plot_training_history(history, exp_dir)
    
    # 7. Run inference on test set
    print("\n" + "=" * 70)
    print("STEP 6: Test Set Evaluation")
    print("=" * 70)
    
    preds, targets, errors, metrics = run_inference(model, test_loader, CONFIG['device'], exp_dir)
    
    # 8. Create preview visualization
    print("\n" + "=" * 70)
    print("STEP 7: Creating Visualizations")
    print("=" * 70)
    
    create_preview_frames(model, test_loader, CONFIG['device'], exp_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {exp_dir}")
    print(f"\nFiles generated:")
    print(f"  - config.json")
    print(f"  - model_best.pt")
    print(f"  - model_final.pt")
    print(f"  - predictions.npz")
    print(f"  - ground_truth.npz")
    print(f"  - errors.npz")
    print(f"  - metrics.json")
    print(f"  - history.json")
    print(f"  - training_history.png")
    print(f"  - predictions_preview.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
physics_inspired_surrogate.py - Physics-Inspired Surrogate for Enthalpy Data

This is NOT a PINN. NOT a PDE-constrained model. NOT a Stefan solver.

This is a DATA-DRIVEN surrogate with SOFT physics-inspired regularization.
"""

import numpy as np
import os
import json
import time
import glob
from datetime import datetime

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
    'data_dir': 'dataset',
    'max_sims_per_type': 400,
    
    # LIGHTWEIGHT ARCHITECTURE
    'features': [8, 16, 32, 64],
    'dropout': 0.05,
    
    'use_T_t': True,
    'use_T_prev': True,
    'use_delta_T': True,
    
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seed': 42,
    
    # SOFT PHYSICS-INSPIRED WEIGHTS
    'regularization_weights': {
        'data': 1.0,
        'temperature_hint': 0.01,
        'spatial_smooth': 0.005,
        'temporal_hint': 0.005,
        'enthalpy_hint': 0.005,
    },
    
    'T_m': 327.5,
    'T_initial': 25.0,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'task_type': 'same_time',
}

# ============================================================================
# UTILITY
# ============================================================================

def create_experiment_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"inspired_surrogate_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
    return exp_dir

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_simulation_files(data_dir, max_sims=None):
    simulations = []
    for bc_name in ["dirichlet", "neumann"]:
        bc_dir = os.path.join(data_dir, bc_name)
        if not os.path.exists(bc_dir):
            continue
        npz_files = sorted(glob.glob(os.path.join(bc_dir, "*.npz")))
        if max_sims:
            npz_files = npz_files[:max_sims]
        for filepath in npz_files:
            data = np.load(filepath)
            simulations.append({
                'T_norm': data['T_norm'],
                'liquid_frac': data['liquid_frac'],
                'times': data['times'],
                'metadata': {'bc_type': bc_name}
            })
    print(f"Loaded {len(simulations)} simulations")
    return simulations

def split_simulations(simulations, train_ratio, val_ratio, test_ratio, seed):
    set_seed(seed)
    n_sims = len(simulations)
    indices = np.random.permutation(n_sims)
    n_train = int(n_sims * train_ratio)
    n_val = int(n_sims * val_ratio)
    n_test = n_sims - n_train - n_val
    train_sims = [simulations[i] for i in indices[:n_train]]
    val_sims = [simulations[i] for i in indices[n_train:n_train + n_val]]
    test_sims = [simulations[i] for i in indices[n_train + n_val:]]
    print(f"Train: {len(train_sims)} | Val: {len(val_sims)} | Test: {len(test_sims)}")
    return train_sims, val_sims, test_sims

# ============================================================================
# DATASET
# ============================================================================

class InspiredDataset(Dataset):
    def __init__(self, simulations, config):
        self.config = config
        self.samples = []
        for sim in simulations:
            T_norm = sim['T_norm']
            phi = sim['liquid_frac']
            n_frames = T_norm.shape[0]
            if n_frames < 2:
                continue
            T_denorm = T_norm * (config['T_m'] - config['T_initial']) + config['T_initial']
            start_idx = 1 if (config['use_T_prev'] or config['use_delta_T']) else 0
            for t in range(start_idx, n_frames):
                channels = []
                if config['use_T_t']:
                    channels.append(T_norm[t])
                if config['use_T_prev']:
                    channels.append(T_norm[t-1])
                if config['use_delta_T']:
                    channels.append(T_norm[t] - T_norm[t-1])
                X = np.stack(channels, axis=0).astype(np.float32)
                y = phi[t].astype(np.float32)
                T_current = T_denorm[t].astype(np.float32)
                T_prev = T_denorm[t-1].astype(np.float32)
                self.samples.append({
                    'X': X, 'y': y, 'T_current': T_current, 'T_prev': T_prev,
                })
        print(f"  Created {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'X': torch.tensor(s['X'], dtype=torch.float32),
            'y': torch.tensor(s['y'], dtype=torch.float32).unsqueeze(0),
            'T_current': torch.tensor(s['T_current'], dtype=torch.float32),
            'T_prev': torch.tensor(s['T_prev'], dtype=torch.float32),
        }

# ============================================================================
# U-NET with CORRECT channel dimensions
# ============================================================================

class DoubleConv(nn.Module):
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
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout)
        )
    
    def forward(self, x):
        return self.mp_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(Up, self).__init__()
        # Use ConvTranspose2d for upsampling
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
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.conv(x))


class InspiredSurrogate(nn.Module):
    """
    U-Net for liquid fraction prediction.
    Input: multi-channel temperature fields
    Output: liquid fraction [0,1]
    """
    def __init__(self, in_channels, n_classes=1, features=[16, 32, 64, 128], dropout=0.05):
        super(InspiredSurrogate, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout)
        
        # Decoder - note the channel dimensions carefully
        # up1: input features[3]*2 (256), output features[3] (128)
        self.up1 = Up(features[3] * 2, features[3], dropout)
        # up2: input features[3] (128) + skip from down2 features[2] (64) = 192, output features[2] (64)
        self.up2 = Up(features[3], features[2], dropout)
        # up3: input features[2] (64) + skip from down1 features[1] (32) = 96, output features[1] (32)
        self.up3 = Up(features[2], features[1], dropout)
        # up4: input features[1] (32) + skip from inc features[0] (16) = 48, output features[0] (16)
        self.up4 = Up(features[1], features[0], dropout)
        
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)           # features[0]
        x2 = self.down1(x1)        # features[1]
        x3 = self.down2(x2)        # features[2]
        x4 = self.down3(x3)        # features[3]
        
        # Bottleneck
        x5 = self.bottleneck(x4)   # features[3] * 2
        
        # Decoder with skip connections
        x = self.up1(x5, x4)       # features[3]
        x = self.up2(x, x3)        # features[2]
        x = self.up3(x, x2)        # features[1]
        x = self.up4(x, x1)        # features[0]
        
        return self.outc(x)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class InspiredLoss(nn.Module):
    def __init__(self, config):
        super(InspiredLoss, self).__init__()
        self.config = config
        self.T_m = config['T_m']
        self.T_initial = config['T_initial']
        
        self.w_data = config['regularization_weights']['data']
        self.w_temp = config['regularization_weights']['temperature_hint']
        self.w_spatial = config['regularization_weights']['spatial_smooth']
        self.w_temporal = config['regularization_weights']['temporal_hint']
        self.w_enthalpy = config['regularization_weights']['enthalpy_hint']
        
        self.mse = nn.MSELoss()
    
    def temperature_hint(self, phi, T_current):
        """Soft: In mushy zone, temperature should be near T_m."""
        mushy = ((phi > 0.1) & (phi < 0.9)).float()
        if mushy.sum() == 0:
            return torch.tensor(0.0, device=phi.device)
        T_dev = torch.abs(T_current - self.T_m) / self.T_m
        return (mushy * T_dev).mean()
    
    def spatial_smoothness(self, phi):
        """Soft: Liquid fraction should be smooth."""
        grad_x = torch.abs(phi[:, :, :, 1:] - phi[:, :, :, :-1])
        grad_y = torch.abs(phi[:, :, 1:, :] - phi[:, :, :-1, :])
        return (grad_x.mean() + grad_y.mean()) / 2
    
    def temporal_hint(self, phi, phi_prev):
        """Soft: Liquid fraction changes gradually."""
        if phi_prev is None:
            return torch.tensor(0.0, device=phi.device)
        return torch.abs(phi - phi_prev).mean()
    
    def enthalpy_hint(self, phi, T_current):
        """Soft: φ should increase with temperature."""
        T_norm = (T_current - self.T_initial) / (self.T_m - self.T_initial)
        T_norm = torch.clamp(T_norm, 0, 1)
        return self.mse(phi, T_norm.unsqueeze(1))
    
    def forward(self, pred, target, T_current, phi_prev=None):
        loss_data = self.mse(pred, target)
        loss_temp = self.temperature_hint(pred, T_current)
        loss_spatial = self.spatial_smoothness(pred)
        loss_temporal = self.temporal_hint(pred, phi_prev)
        loss_enthalpy = self.enthalpy_hint(pred, T_current)
        
        total = (
            self.w_data * loss_data +
            self.w_temp * loss_temp +
            self.w_spatial * loss_spatial +
            self.w_temporal * loss_temporal +
            self.w_enthalpy * loss_enthalpy
        )
        
        self.components = {
            'data': loss_data.item(),
            'temp_hint': loss_temp.item(),
            'spatial': loss_spatial.item(),
            'temporal': loss_temporal.item(),
            'enthalpy': loss_enthalpy.item(),
        }
        return total

# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    components = {k: 0.0 for k in ['data', 'temp_hint', 'spatial', 'temporal', 'enthalpy']}
    all_preds, all_targets = [], []
    phi_prev = None
    
    for batch in loader:
        X = batch['X'].to(device)
        y = batch['y'].to(device)
        T_current = batch['T_current'].unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y, T_current, phi_prev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k in components:
            components[k] += criterion.components.get(k, 0)
        
        all_preds.append(y_pred.detach().cpu().numpy())
        all_targets.append(y.cpu().numpy())
        phi_prev = y_pred.detach()
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mae = mean_absolute_error(targets.flatten(), preds.flatten())
    
    n = len(loader)
    return total_loss / n, mae, {k: v/n for k, v in components.items()}


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            T_current = batch['T_current'].unsqueeze(1).to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y, T_current)
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mae = mean_absolute_error(targets.flatten(), preds.flatten())
    return total_loss / len(loader), mae

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("PHYSICS-INSPIRED SURROGATE MODEL")
   
    
    set_seed(CONFIG['seed'])
    exp_dir = create_experiment_dir()
    
    sims = load_simulation_files(CONFIG['data_dir'], CONFIG['max_sims_per_type'])
    if len(sims) == 0:
        print("No data found!")
        return
    
    train_sims, val_sims, test_sims = split_simulations(
        sims, CONFIG['train_ratio'], CONFIG['val_ratio'], 
        CONFIG['test_ratio'], CONFIG['seed']
    )
    
    train_ds = InspiredDataset(train_sims, CONFIG)
    val_ds = InspiredDataset(val_sims, CONFIG)
    test_ds = InspiredDataset(test_sims, CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    sample = train_ds[0]
    n_channels = sample['X'].shape[0]
    
    model = InspiredSurrogate(
        in_channels=n_channels,
        features=CONFIG['features'],
        dropout=CONFIG['dropout']
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Input channels: {n_channels}")
    print(f"Features: {CONFIG['features']}")
    
    criterion = InspiredLoss(CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    device = CONFIG['device']
    model = model.to(device)
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print(f"Primary: Data loss (weight={CONFIG['regularization_weights']['data']})")
    print(f"Soft hints: temperature({CONFIG['regularization_weights']['temperature_hint']}), "
          f"spatial({CONFIG['regularization_weights']['spatial_smooth']}), "
          f"enthalpy({CONFIG['regularization_weights']['enthalpy_hint']})")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_mae, comp = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model_best.pt'))
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"MAE: {train_mae:.4f}/{val_mae:.4f} | Data: {comp['data']:.4f}")
    
    # Test
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'model_best.pt')))
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y_pred = model(X).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(batch['y'].numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    metrics = {
        'MSE': mean_squared_error(targets.flatten(), preds.flatten()),
        'MAE': mean_absolute_error(targets.flatten(), preds.flatten()),
        'RMSE': np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())),
        'R2': r2_score(targets.flatten(), preds.flatten()),
    }
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 70)
    print(f"Results saved to: {exp_dir}")
if __name__ == "__main__":
    main()
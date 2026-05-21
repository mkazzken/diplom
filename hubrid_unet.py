"""
physics_inspired_surrogate.py - Physics-Inspired Surrogate for Enthalpy Data

This is NOT a PINN. NOT a PDE-constrained model. NOT a Stefan solver.

This is a DATA-DRIVEN surrogate with SOFT physics-inspired regularization.

Outputs (matching second script):
    config.json, model_best.pt, model_final.pt,
    predictions.npz, ground_truth.npz, errors.npz,
    metrics.json, history.json,
    training_history.png, predictions_preview.png
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
    train_sims = [simulations[i] for i in indices[:n_train]]
    val_sims   = [simulations[i] for i in indices[n_train:n_train + n_val]]
    test_sims  = [simulations[i] for i in indices[n_train + n_val:]]
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
                    channels.append(T_norm[t - 1])
                if config['use_delta_T']:
                    channels.append(T_norm[t] - T_norm[t - 1])
                X = np.stack(channels, axis=0).astype(np.float32)
                y = phi[t].astype(np.float32)
                T_current = T_denorm[t].astype(np.float32)
                T_prev    = T_denorm[t - 1].astype(np.float32)
                self.samples.append({
                    'X': X, 'y': y,
                    'T_current': T_current, 'T_prev': T_prev,
                })
        print(f"  Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'X':         torch.tensor(s['X'],         dtype=torch.float32),
            'y':         torch.tensor(s['y'],         dtype=torch.float32).unsqueeze(0),
            'T_current': torch.tensor(s['T_current'], dtype=torch.float32),
            'T_prev':    torch.tensor(s['T_prev'],    dtype=torch.float32),
        }

# ============================================================================
# U-NET
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.mp_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout))

    def forward(self, x):
        return self.mp_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv    = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class InspiredSurrogate(nn.Module):
    """U-Net for liquid fraction prediction."""
    def __init__(self, in_channels, n_classes=1, features=[16, 32, 64, 128], dropout=0.05):
        super().__init__()
        self.inc        = DoubleConv(in_channels,       features[0],     dropout)
        self.down1      = Down(features[0],              features[1],     dropout)
        self.down2      = Down(features[1],              features[2],     dropout)
        self.down3      = Down(features[2],              features[3],     dropout)
        self.bottleneck = DoubleConv(features[3],        features[3] * 2, dropout)
        self.up1        = Up(features[3] * 2,            features[3],     dropout)
        self.up2        = Up(features[3],                features[2],     dropout)
        self.up3        = Up(features[2],                features[1],     dropout)
        self.up4        = Up(features[1],                features[0],     dropout)
        self.outc       = OutConv(features[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class InspiredLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T_m       = config['T_m']
        self.T_initial = config['T_initial']
        self.w_data     = config['regularization_weights']['data']
        self.w_temp     = config['regularization_weights']['temperature_hint']
        self.w_spatial  = config['regularization_weights']['spatial_smooth']
        self.w_temporal = config['regularization_weights']['temporal_hint']
        self.w_enthalpy = config['regularization_weights']['enthalpy_hint']
        self.mse = nn.MSELoss()

    def temperature_hint(self, phi, T_current):
        mushy = ((phi > 0.1) & (phi < 0.9)).float()
        if mushy.sum() == 0:
            return torch.tensor(0.0, device=phi.device)
        T_dev = torch.abs(T_current - self.T_m) / self.T_m
        return (mushy * T_dev).mean()

    def spatial_smoothness(self, phi):
        grad_x = torch.abs(phi[:, :, :, 1:] - phi[:, :, :, :-1])
        grad_y = torch.abs(phi[:, :, 1:, :] - phi[:, :, :-1, :])
        return (grad_x.mean() + grad_y.mean()) / 2

    def temporal_hint(self, phi, phi_prev):
        if phi_prev is None:
            return torch.tensor(0.0, device=phi.device)
        return torch.abs(phi - phi_prev).mean()

    def enthalpy_hint(self, phi, T_current):
        T_norm = (T_current - self.T_initial) / (self.T_m - self.T_initial)
        T_norm = torch.clamp(T_norm, 0, 1)
        return self.mse(phi, T_norm.unsqueeze(1))

    def forward(self, pred, target, T_current, phi_prev=None):
        loss_data     = self.mse(pred, target)
        loss_temp     = self.temperature_hint(pred, T_current)
        loss_spatial  = self.spatial_smoothness(pred)
        loss_temporal = self.temporal_hint(pred, phi_prev)
        loss_enthalpy = self.enthalpy_hint(pred, T_current)

        total = (
            self.w_data     * loss_data     +
            self.w_temp     * loss_temp     +
            self.w_spatial  * loss_spatial  +
            self.w_temporal * loss_temporal +
            self.w_enthalpy * loss_enthalpy
        )
        self.components = {
            'data':     loss_data.item(),
            'temp_hint': loss_temp.item(),
            'spatial':  loss_spatial.item(),
            'temporal': loss_temporal.item(),
            'enthalpy': loss_enthalpy.item(),
        }
        return total

# ============================================================================
# METRICS
# ============================================================================

def dice_coefficient(y_true, y_pred, threshold=0.5):
    true_bin = (y_true >= threshold).astype(np.float32)
    pred_bin = (y_pred >= threshold).astype(np.float32)
    intersection = np.sum(true_bin * pred_bin)
    union = np.sum(true_bin) + np.sum(pred_bin)
    if union == 0:
        return 1.0
    return float(2 * intersection / union)


def compute_all_metrics(y_true, y_pred):
    flat_t, flat_p = y_true.flatten(), y_pred.flatten()
    return {
        'MSE':  float(mean_squared_error(flat_t, flat_p)),
        'MAE':  float(mean_absolute_error(flat_t, flat_p)),
        'RMSE': float(np.sqrt(mean_squared_error(flat_t, flat_p))),
        'R2':   float(r2_score(flat_t, flat_p)),
        'Dice': dice_coefficient(y_true, y_pred),
    }

# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = None
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
            self.counter   = 0
        return self.early_stop

# ============================================================================
# TRAINING LOOPS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    comp_acc   = {k: 0.0 for k in ['data', 'temp_hint', 'spatial', 'temporal', 'enthalpy']}
    all_preds, all_targets = [], []
    phi_prev = None

    for batch in loader:
        X         = batch['X'].to(device)
        y         = batch['y'].to(device)
        T_current = batch['T_current'].unsqueeze(1).to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss   = criterion(y_pred, y, T_current, phi_prev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in comp_acc:
            comp_acc[k] += criterion.components.get(k, 0)

        all_preds.append(y_pred.detach().cpu().numpy())
        all_targets.append(y.cpu().numpy())
        phi_prev = y_pred.detach()

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(targets, preds)
    n = len(loader)
    return total_loss / n, metrics, {k: v / n for k, v in comp_acc.items()}


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            X         = batch['X'].to(device)
            y         = batch['y'].to(device)
            T_current = batch['T_current'].unsqueeze(1).to(device)
            y_pred    = model(X)
            loss      = criterion(y_pred, y, T_current)
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(targets, preds)
    return total_loss / len(loader), metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, exp_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'],   'r-', label='Val Loss',   linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    train_mae = [m['MAE'] for m in history['train_metrics']]
    val_mae   = [m['MAE'] for m in history['val_metrics']]
    axes[0, 1].plot(train_mae, 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(val_mae,   'r-', label='Val MAE',   linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    train_dice = [m['Dice'] for m in history['train_metrics']]
    val_dice   = [m['Dice'] for m in history['val_metrics']]
    axes[1, 0].plot(train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1, 0].plot(val_dice,   'r-', label='Val Dice',   linewidth=2)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].set_title('Interface Dice Coefficient')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_history.png'), dpi=150)
    plt.close()
    print("Training history plot saved.")


def create_preview_frames(model, test_loader, device, exp_dir, n_samples=4):
    model.eval()
    batch    = next(iter(test_loader))
    X_batch  = batch['X'][:n_samples].to(device)
    y_batch  = batch['y'][:n_samples]

    with torch.no_grad():
        y_pred = model(X_batch).cpu().numpy()

    y_true  = y_batch.numpy()
    X_input = X_batch.cpu().numpy()
    actual_n = min(n_samples, X_input.shape[0])

    fig, axes = plt.subplots(actual_n, 4, figsize=(16, 4 * actual_n))
    if actual_n == 1:
        axes = axes.reshape(1, -1)

    for i in range(actual_n):
        axes[i, 0].imshow(X_input[i, 0], cmap='hot',     origin='lower')
        axes[i, 0].set_title('Input: Current Temperature'); axes[i, 0].axis('off')

        axes[i, 1].imshow(y_true[i, 0],  cmap='coolwarm', origin='lower', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth'); axes[i, 1].axis('off')

        axes[i, 2].imshow(y_pred[i, 0],  cmap='coolwarm', origin='lower', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction (MAE: {np.mean(np.abs(y_true[i,0]-y_pred[i,0])):.4f})')
        axes[i, 2].axis('off')

        error = np.abs(y_true[i, 0] - y_pred[i, 0])
        im = axes[i, 3].imshow(error, cmap='Reds', origin='lower', vmin=0, vmax=0.5)
        axes[i, 3].set_title('Error Map'); axes[i, 3].axis('off')

    plt.colorbar(im, ax=axes[:, 3], shrink=0.5)
    plt.suptitle('Physics-Inspired Surrogate Predictions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'predictions_preview.png'), dpi=150)
    plt.close()
    print("Predictions preview saved.")

# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, test_loader, device, exp_dir):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            X      = batch['X'].to(device)
            y_pred = model(X).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(batch['y'].numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    errors  = np.abs(targets - preds)

    np.savez_compressed(os.path.join(exp_dir, 'predictions.npz'),  arr=preds)
    np.savez_compressed(os.path.join(exp_dir, 'ground_truth.npz'), arr=targets)
    np.savez_compressed(os.path.join(exp_dir, 'errors.npz'),       arr=errors)

    metrics = compute_all_metrics(targets, preds)
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 70)

    return preds, targets, errors, metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHYSICS-INSPIRED SURROGATE MODEL")
    print("Soft physics regularization. Data-driven U-Net surrogate.")
    print("=" * 70)

    set_seed(CONFIG['seed'])
    exp_dir = create_experiment_dir()
    print(f"\nExperiment directory: {exp_dir}")

    # 1. Load data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Dataset")
    print("=" * 70)
    sims = load_simulation_files(CONFIG['data_dir'], CONFIG['max_sims_per_type'])
    if len(sims) == 0:
        print("No data found! Check dataset path.")
        return

    # 2. Split
    print("\n" + "=" * 70)
    print("STEP 2: Splitting Data (By Simulation — No Leakage)")
    print("=" * 70)
    train_sims, val_sims, test_sims = split_simulations(
        sims, CONFIG['train_ratio'], CONFIG['val_ratio'],
        CONFIG['test_ratio'], CONFIG['seed']
    )

    # 3. Datasets & loaders
    print("\n" + "=" * 70)
    print("STEP 3: Creating Datasets")
    print("=" * 70)
    train_ds = InspiredDataset(train_sims, CONFIG)
    val_ds   = InspiredDataset(val_sims,   CONFIG)
    test_ds  = InspiredDataset(test_sims,  CONFIG)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # 4. Model
    print("\n" + "=" * 70)
    print("STEP 4: Creating Model")
    print("=" * 70)
    n_channels = train_ds[0]['X'].shape[0]
    model = InspiredSurrogate(
        in_channels=n_channels,
        features=CONFIG['features'],
        dropout=CONFIG['dropout'],
    )
    print(f"Model: Physics-Inspired U-Net")
    print(f"Input channels: {n_channels} | Features: {CONFIG['features']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion      = InspiredLoss(CONFIG)
    optimizer      = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                                weight_decay=CONFIG['weight_decay'])
    scheduler      = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])

    device = CONFIG['device']
    model  = model.to(device)

    # 5. Training
    print("\n" + "=" * 70)
    print("STEP 5: Training")
    print(f"Primary loss weight : {CONFIG['regularization_weights']['data']}")
    print(f"Soft hints — temp: {CONFIG['regularization_weights']['temperature_hint']}, "
          f"spatial: {CONFIG['regularization_weights']['spatial_smooth']}, "
          f"enthalpy: {CONFIG['regularization_weights']['enthalpy_hint']}")
    print("=" * 70)

    best_val_loss = float('inf')
    start_time    = time.time()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': [],
        'learning_rates': [],
    }
    val_loss    = float('inf')
    val_metrics = {}

    for epoch in range(CONFIG['epochs']):
        train_loss, train_metrics, comp = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics           = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': val_metrics,
            }, os.path.join(exp_dir, 'model_best.pt'))

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"MAE: {train_metrics['MAE']:.4f}/{val_metrics['MAE']:.4f} | "
                  f"Dice: {train_metrics['Dice']:.4f}/{val_metrics['Dice']:.4f} | "
                  f"Data: {comp['data']:.4f} | "
                  f"Time: {elapsed:.0f}s")

    # Save final model
    torch.save({
        'epoch': CONFIG['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'metrics': val_metrics,
    }, os.path.join(exp_dir, 'model_final.pt'))

    # Save history
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        serializable = {}
        for key, values in history.items():
            if key in ['train_metrics', 'val_metrics']:
                serializable[key] = [{k: float(v) for k, v in m.items()} for m in values]
            else:
                serializable[key] = [float(v) for v in values]
        json.dump(serializable, f, indent=4)

    print(f"\nTraining completed in {time.time() - start_time:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # 6. Plot training history
    print("\n" + "=" * 70)
    print("STEP 6: Plotting Training History")
    print("=" * 70)
    plot_training_history(history, exp_dir)

    # 7. Test inference
    print("\n" + "=" * 70)
    print("STEP 7: Test Set Evaluation")
    print("=" * 70)
    best_ckpt = torch.load(os.path.join(exp_dir, 'model_best.pt'))
    model.load_state_dict(best_ckpt['model_state_dict'])
    run_inference(model, test_loader, device, exp_dir)

    # 8. Predictions preview
    print("\n" + "=" * 70)
    print("STEP 8: Creating Predictions Preview")
    print("=" * 70)
    create_preview_frames(model, test_loader, device, exp_dir)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {exp_dir}")
    print("\nFiles generated:")
    for fname in [
        'config.json', 'model_best.pt', 'model_final.pt',
        'predictions.npz', 'ground_truth.npz', 'errors.npz',
        'metrics.json', 'history.json',
        'training_history.png', 'predictions_preview.png',
    ]:
        print(f"  - {fname}")
    print("=" * 70)


if __name__ == "__main__":
    main()
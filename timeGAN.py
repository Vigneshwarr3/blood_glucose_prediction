import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""
    def __init__(self, data, seq_len=24):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            seq_len: length of sequences to generate
        """
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len + 1
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.seq_len])

def prepare_data(df, seq_len=24, features=None):
    """
    Prepare time series data for GAN training
    
    Args:
        df: DataFrame with timestamp and all features
        seq_len: sequence length for training
        features: list of feature columns to use (if None, uses all numeric)
    """
    # Select features
    if features is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove timestamp if it's numeric
        if 'timestamp' in numeric_cols:
            numeric_cols.remove('timestamp')
        features = numeric_cols
    else:
        # Filter to only include features that exist in the DataFrame
        features = [f for f in features if f in df.columns]
    
    # Extract feature data
    data = df[features].values
    
    # Handle missing values (forward fill, then backward fill, then fill with 0)
    df_features = pd.DataFrame(data, columns=features)
    df_features = df_features.fillna(method='ffill', limit=10).fillna(method='bfill', limit=10).fillna(0)
    data = df_features.values
    
    # Normalize data to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    dataset = TimeSeriesDataset(data_scaled, seq_len)
    
    return dataset, scaler, features
# ============================================================================
# 2. GENERATOR NETWORK
# ============================================================================

class Generator(nn.Module):
    """
    Generator network for TimeGAN
    Uses LSTM to generate realistic time series sequences
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None):
        super(Generator, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
            
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1], will be scaled to [0, 1]
        )
        
    def forward(self, noise, hidden=None):
        """
        Args:
            noise: random noise tensor (batch_size, seq_len, noise_dim)
            hidden: hidden state tuple (h_n, c_n)
        """
        # Embed noise
        embedded = self.embedding(noise)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Generate output
        output = self.fc(lstm_out)
        
        # Scale from [-1, 1] to [0, 1]
        output = (output + 1) / 2
        
        return output, hidden

# ============================================================================
# 3. DISCRIMINATOR NETWORK
# ============================================================================

class Discriminator(nn.Module):
    """
    Discriminator network for TimeGAN
    Uses LSTM to distinguish real from fake time series
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(Discriminator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: time series tensor (batch_size, seq_len, input_dim)
        Returns:
            probability that input is real
        """
        # Embed input
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)
        
        # Use last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classify
        output = self.fc(last_output)
        
        return output

# ============================================================================
# 4. SUPERVISOR NETWORK (for TimeGAN)
# ============================================================================

class Supervisor(nn.Module):
    """
    Supervisor network helps generator learn temporal patterns
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(Supervisor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# ============================================================================
# 5. TIME SERIES GAN MODEL
# ============================================================================

class TimeSeriesGAN:
    """
    Complete Time Series GAN implementation
    """
    def __init__(self, input_dim, noise_dim=10, hidden_dim=128, num_layers=2, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        
        # Initialize networks
        # Generator needs noise_dim and output_dim (which is input_dim)
        self.generator = Generator(noise_dim, input_dim, hidden_dim, num_layers).to(self.device)
        self.discriminator = Discriminator(input_dim, hidden_dim, num_layers).to(self.device)
        self.supervisor = Supervisor(input_dim, hidden_dim, num_layers).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.s_optimizer = optim.Adam(self.supervisor.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def train(self, dataloader, epochs=100, seq_len=24, d_steps=1, g_steps=1):
        """
        Train the Time Series GAN
        
        Args:
            dataloader: DataLoader with real time series sequences
            epochs: number of training epochs
            seq_len: sequence length
            d_steps: discriminator training steps per epoch
            g_steps: generator training steps per epoch
        """
        self.generator.train()
        self.discriminator.train()
        self.supervisor.train()
        
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            s_losses = []
            
            for batch_idx, real_data in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # ============================================
                # Train Discriminator
                # ============================================
                for _ in range(d_steps):
                    self.d_optimizer.zero_grad()
                    
                    # Real data
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    real_output = self.discriminator(real_data)
                    d_loss_real = self.criterion(real_output, real_labels)
                    
                    # Fake data
                    noise = torch.randn(batch_size, seq_len, self.noise_dim).to(self.device)
                    fake_data, _ = self.generator(noise)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                    fake_output = self.discriminator(fake_data.detach())
                    d_loss_fake = self.criterion(fake_output, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    d_losses.append(d_loss.item())
                
                # ============================================
                # Train Generator
                # ============================================
                for _ in range(g_steps):
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake data
                    noise = torch.randn(batch_size, seq_len, self.noise_dim).to(self.device)
                    fake_data, _ = self.generator(noise)
                    
                    # Try to fool discriminator
                    fake_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_output = self.discriminator(fake_data)
                    g_loss = self.criterion(fake_output, fake_labels)
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                
                # ============================================
                # Train Supervisor (optional, for better temporal patterns)
                # ============================================
                if batch_idx % 5 == 0:  # Train supervisor less frequently
                    self.s_optimizer.zero_grad()
                    supervised_output = self.supervisor(real_data)
                    s_loss = self.mse_loss(supervised_output, real_data)
                    s_loss.backward()
                    self.s_optimizer.step()
                    s_losses.append(s_loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_d_loss = np.mean(d_losses) if d_losses else 0
                avg_g_loss = np.mean(g_losses) if g_losses else 0
                avg_s_loss = np.mean(s_losses) if s_losses else 0
                print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {avg_d_loss:.4f} | "
                      f"G_loss: {avg_g_loss:.4f} | S_loss: {avg_s_loss:.4f}")
    
    def generate(self, n_samples, seq_len=24):
        """
        Generate synthetic time series data
        
        Args:
            n_samples: number of sequences to generate
            seq_len: length of each sequence
        Returns:
            numpy array of generated sequences
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, seq_len, self.noise_dim).to(self.device)
            fake_data, _ = self.generator(noise)
            fake_data = fake_data.cpu().numpy()
        
        return fake_data
    
    def save(self, filepath):
        """Save model checkpoints"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'supervisor': self.supervisor.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoints"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.supervisor.load_state_dict(checkpoint['supervisor'])

# ============================================================================
# 6. COMPLETE PIPELINE
# ============================================================================

def train_time_series_gan(df, features=None, seq_len=24, epochs=100, 
                          batch_size=64, hidden_dim=128, num_layers=2):
    """
    Complete pipeline to train Time Series GAN
    
    Args:
        df: DataFrame with timestamp and all features
        features: list of feature columns (None = all numeric)
        seq_len: sequence length for training
        epochs: number of training epochs
        batch_size: batch size for training
        hidden_dim: hidden dimension for LSTM
        num_layers: number of LSTM layers
    """
    print("=" * 80)
    print("TIME SERIES GAN TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Prepare data
    print("\n[1/4] Preparing data...")
    dataset, scaler, feature_names = prepare_data(df, seq_len, features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Features: {feature_names}")
    print(f"   Sequence length: {seq_len}")
    
    # 2. Initialize GAN
    print("\n[2/4] Initializing Time Series GAN...")
    input_dim = len(feature_names)
    gan = TimeSeriesGAN(
        input_dim=input_dim,
        noise_dim=10,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"   Device: {gan.device}")
    print(f"   Input dimension: {input_dim}")
    
    # 3. Train GAN
    print("\n[3/4] Training GAN...")
    gan.train(dataloader, epochs=epochs, seq_len=seq_len)
    
    # 4. Save model
    print("\n[4/4] Saving model...")
    gan.save('time_series_gan_model.pt')
    print("   Model saved to 'time_series_gan_model.pt'")
    
    return gan, scaler, feature_names

def generate_synthetic_data(gan, scaler, feature_names, n_samples=100, seq_len=24):
    """
    Generate synthetic patient data
    
    Args:
        gan: trained TimeSeriesGAN model
        scaler: MinMaxScaler used for normalization
        feature_names: list of feature names
        n_samples: number of sequences to generate
        seq_len: length of each sequence
    Returns:
        DataFrame with synthetic data
    """
    # Generate synthetic sequences
    synthetic_data = gan.generate(n_samples, seq_len)
    
    # Reshape to (n_samples * seq_len, n_features)
    synthetic_flat = synthetic_data.reshape(-1, len(feature_names))
    
    # Inverse transform to original scale
    synthetic_original = scaler.inverse_transform(synthetic_flat)
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(synthetic_original, columns=feature_names)
    
    # Add timestamp (assuming 5-minute intervals)
    timestamps = pd.date_range(
        start='2021-01-01 00:00:00',
        periods=len(df_synthetic),
        freq='5min'
    )
    df_synthetic.insert(0, 'timestamp', timestamps)
    
    return df_synthetic

# ============================================================================
# 7. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load your data (from the CSV you created earlier)
    df = pd.read_csv('data/train/559-ws-training_all_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Select features to generate based on actual columns in the CSV
    # Exclude non-numeric columns: timestamp, patient, event
    numeric_features = [
        'glucose_level', 'finger_stick', 'basal', 'temp_basal', 'bolus', 'meal',
        'sleep', 'work', 'stressors', 'hypo_event', 'illness', 'exercise',
        'basis_heart_rate', 'basis_gsr', 'basis_skin_temperature',
        'basis_air_temperature', 'basis_steps', 'basis_sleep'
    ]
    
    # Filter to only include features that actually exist in the DataFrame
    features = [f for f in numeric_features if f in df.columns]
    
    print(f"Using {len(features)} features: {features}")
    
    # Train GAN
    gan, scaler, feature_names = train_time_series_gan(
        df,
        features=features,
        seq_len=24,  # 24 timesteps = 2 hours (5-min intervals)
        epochs=200,
        batch_size=64,
        hidden_dim=128,
        num_layers=2
    )
    
    # Generate synthetic data
    print("\n" + "=" * 80)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 80)
    
    df_synthetic = generate_synthetic_data(
        gan, scaler, feature_names,
        n_samples=100,  # Generate 100 sequences
        seq_len=24
    )
    
    # Save synthetic data
    df_synthetic.to_csv('synthetic_patient_data.csv', index=False)
    print(f"\n✅ Generated {len(df_synthetic)} synthetic data points")
    print("   Saved to 'synthetic_patient_data.csv'")
    
    # Visualize comparison
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Glucose comparison
    if 'glucose_level' in df.columns and 'glucose_level' in df_synthetic.columns:
        axes[0].plot(df['glucose_level'].head(500).dropna(), label='Real', alpha=0.7)
        axes[0].plot(df_synthetic['glucose_level'].head(500), label='Synthetic', alpha=0.7)
        axes[0].set_title('Glucose Level: Real vs Synthetic')
        axes[0].legend()
        axes[0].set_ylabel('Glucose (mg/dL)')
    
    # Heart rate comparison
    if 'basis_heart_rate' in df.columns and 'basis_heart_rate' in df_synthetic.columns:
        real_hr = df['basis_heart_rate'].head(500).dropna()
        if len(real_hr) > 0:
            axes[1].plot(real_hr, label='Real', alpha=0.7)
            axes[1].plot(df_synthetic['basis_heart_rate'].head(500), label='Synthetic', alpha=0.7)
            axes[1].set_title('Heart Rate: Real vs Synthetic')
            axes[1].legend()
            axes[1].set_ylabel('Heart Rate (bpm)')
    
    # Steps comparison
    if 'basis_steps' in df.columns and 'basis_steps' in df_synthetic.columns:
        real_steps = df['basis_steps'].head(500).dropna()
        if len(real_steps) > 0:
            axes[2].plot(real_steps, label='Real', alpha=0.7)
            axes[2].plot(df_synthetic['basis_steps'].head(500), label='Synthetic', alpha=0.7)
            axes[2].set_title('Steps: Real vs Synthetic')
            axes[2].legend()
            axes[2].set_ylabel('Steps')
            axes[2].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('real_vs_synthetic_comparison.png', dpi=150)
    print("\n✅ Visualization saved to 'real_vs_synthetic_comparison.png'")
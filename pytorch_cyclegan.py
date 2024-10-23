import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio
import itertools
import kagglehub
import os

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(channels),
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial Convolution Block
        model = [
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        model += [
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True)
        ]
        
        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResBlock(256)]
        
        # Upsampling
        model += [
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Output Layer
        model += [
            nn.Conv1d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        def conv_block(in_filters, out_filters, stride):
            layers = [
                nn.Conv1d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm1d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers
        
        self.model = nn.Sequential(
            *conv_block(input_channels, 64, 2),
            *conv_block(64, 128, 2),
            *conv_block(128, 256, 2),
            *conv_block(256, 512, 1),
            nn.Conv1d(512, 1, kernel_size=4, stride=1, padding=1)  # Output PatchGAN
        )
    
    def forward(self, x):
        return self.model(x)

class CycleGANLosses:
    def __init__(self, device):
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.device = device
    
    def gan_loss(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion_gan(pred, target.to(self.device))
    
    def cycle_loss(self, real, reconstructed, lambda_cycle=10.0):
        return lambda_cycle * self.criterion_cycle(reconstructed, real)
    
    def identity_loss(self, real, same, lambda_identity=5.0):
        return lambda_identity * self.criterion_identity(same, real)

def train_cyclegan(dataloader, G_A2B, G_B2A, D_A, D_B, optimizer_G, optimizer_D, loss_fn, num_epochs=100):
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()

    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Train Generators A2B and B2A
            optimizer_G.zero_grad()
            
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)
            
            loss_G_A2B = loss_fn.gan_loss(D_B(fake_B), True)
            loss_G_B2A = loss_fn.gan_loss(D_A(fake_A), True)
            loss_cycle_A = loss_fn.cycle_loss(real_A, rec_A)
            loss_cycle_B = loss_fn.cycle_loss(real_B, rec_B)
            
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminators A and B
            optimizer_D.zero_grad()

            loss_D_A = loss_fn.gan_loss(D_A(real_A), True) + loss_fn.gan_loss(D_A(fake_A.detach()), False)
            loss_D_B = loss_fn.gan_loss(D_B(real_B), True) + loss_fn.gan_loss(D_B(fake_B.detach()), False)

            loss_D = (loss_D_A + loss_D_B) / 2
            loss_D.backward()
            optimizer_D.step()

        print(f"Epoch [{epoch}/{num_epochs}] Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}")

class AudioDataset(Dataset):
    def __init__(self, audio_files_A, audio_files_B, transform=None):
        self.audio_files_A = audio_files_A
        self.audio_files_B = audio_files_B
        self.transform = transform
    
    def __len__(self):
        return min(len(self.audio_files_A), len(self.audio_files_B))
    
    def __getitem__(self, idx):
        audio_A, _ = torchaudio.load(self.audio_files_A[idx])
        audio_B, _ = torchaudio.load(self.audio_files_B[idx])
        
        if self.transform:
            audio_A = self.transform(audio_A)
            audio_B = self.transform(audio_B)
        
        return audio_A, audio_B

# Dataset class for Cats and Dogs audio
class AudioDataset(Dataset):
    def __init__(self, audio_files_A, audio_files_B, transform=None):
        self.audio_files_A = audio_files_A
        self.audio_files_B = audio_files_B
        self.transform = transform
    
    def __len__(self):
        return min(len(self.audio_files_A), len(self.audio_files_B))
    
    def __getitem__(self, idx):
        audio_A, _ = torchaudio.load(self.audio_files_A[idx])  # Load cat audio
        audio_B, _ = torchaudio.load(self.audio_files_B[idx])  # Load dog audio
        
        if self.transform:
            audio_A = self.transform(audio_A)
            audio_B = self.transform(audio_B)
        
        return audio_A, audio_B

if __name__ == "__main__":
    # Define paths for the dataset
    data_dir = "data"
    train_cat_audio_dir = os.path.join(data_dir, "cats_dogs", "train", "cat")
    train_dog_audio_dir = os.path.join(data_dir, "cats_dogs", "train", "dog")
    test_cat_audio_dir = os.path.join(data_dir,  "cats_dogs", "test", "cat")
    test_dog_audio_dir = os.path.join(data_dir,  "cats_dogs", "test", "dog")

    # Check if these directories exist
    for dir_path in [train_cat_audio_dir, train_dog_audio_dir, test_cat_audio_dir, test_dog_audio_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    # Create a list of file paths for cats and dogs for both training and testing
    cat_audio_files = (
        [os.path.join(train_cat_audio_dir, f) for f in os.listdir(train_cat_audio_dir) if f.endswith(".wav")] +
        [os.path.join(test_cat_audio_dir, f) for f in os.listdir(test_cat_audio_dir) if f.endswith(".wav")]
    )

    dog_audio_files = (
        [os.path.join(train_dog_audio_dir, f) for f in os.listdir(train_dog_audio_dir) if f.endswith(".wav")] +
        [os.path.join(test_dog_audio_dir, f) for f in os.listdir(test_dog_audio_dir) if f.endswith(".wav")]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    G_A2B = Generator().to(device)  # Cat -> Dog
    G_B2A = Generator().to(device)  # Dog -> Cat
    D_A = Discriminator().to(device)  # Discriminator for cats
    D_B = Discriminator().to(device)  # Discriminator for dogs
    
    optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
    
    loss_fn = CycleGANLosses(device)
    
    # Create the dataset and dataloader
    dataset = AudioDataset(
        audio_files_A=cat_audio_files,
        audio_files_B=dog_audio_files,
        transform=transforms.Compose([transforms.Normalize(mean=[0], std=[1])])
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Start training CycleGAN
    train_cyclegan(
        dataloader, 
        G_A2B, G_B2A, D_A, D_B, 
        optimizer_G, optimizer_D, 
        loss_fn, num_epochs=100
    )

    
   

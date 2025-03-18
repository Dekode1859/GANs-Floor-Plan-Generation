import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import os

# Utility functions - removing display, only saving to disk
def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), name='image.png'):
    path = "floor_runs/"
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # naming convention: floors/runs/epoch_step.png
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.savefig(path + name)
    plt.close()

# Discriminator implementation
class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Contracting path
        self.contract1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.contract2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.contract3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.contract4 = nn.Sequential(
            nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification layer
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.contract4(x)
        x = self.final(x)
        return x

# Loss functions and parameters
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 100

# Hyperparameters
n_epochs = 100
input_dim = 3
real_dim = 3
display_step = 100
batch_size = 4
lr = 0.0002
target_shape = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Early stopping parameters
patience = 10
min_delta = 0.001

# Initialize dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Folder path should have a train and test folder with the images inside them
dataset = torchvision.datasets.ImageFolder("floors", transform=transform)

# Generator Loss Function
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    # Generate fake image
    fake = gen(condition)
    
    # Get adversarial loss
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    
    # Get reconstruction loss
    gen_rec_loss = recon_criterion(real, fake)
    
    # Combine losses
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    return gen_loss, fake

# Initialize generator using pre-trained UNet from smp
gen = smp.Unet(
    encoder_name="resnet34",  # Choose any encoder from smp's available encoders
    encoder_weights="imagenet", # Use pre-trained weights
    in_channels=input_dim,
    classes=real_dim,
    activation="sigmoid"  # Using sigmoid for output in [0,1] range
).to(device)

# Initialize discriminator
disc = Discriminator(input_dim + real_dim).to(device)

# Initialize optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
    
    def __call__(self, val_loss, model_dict):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model
            torch.save(model_dict, self.checkpoint_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# Training loop
def train(save_model=True):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=patience, 
        min_delta=min_delta,
        checkpoint_path="best_pix2pix_smp.pth"
    )
    
    best_generator_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training loop
        running_generator_loss = 0.0
        running_discriminator_loss = 0.0
        n_batches = 0
        
        for image, _ in tqdm(dataloader):
            # Process images
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            condition = nn.functional.interpolate(condition, size=(target_shape, target_shape), mode='bilinear')
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=(target_shape, target_shape), mode='bilinear')
            
            # Move to device
            condition = condition.to(device)
            real = real.to(device)
            
            # TRAINING DISCRIMINATOR
            disc_opt.zero_grad()
            
            # Train with fake images
            with torch.no_grad():
                fake = gen(condition)
            
            disc_fake_hat = disc(fake.detach(), condition)
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            
            # Train with real images
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
            
            # Combine discriminator losses
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            disc_opt.step()
            
            # TRAINING GENERATOR
            gen_opt.zero_grad()
            
            # Generate fake image and get losses
            gen_loss, fake = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward()
            gen_opt.step()
            
            # Keep track of average losses
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step
            
            # Keep track of epoch losses
            running_generator_loss += gen_loss.item()
            running_discriminator_loss += disc_loss.item()
            n_batches += 1
            
            # Visualization and logging
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"\nEpoch {epoch}: Step {cur_step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Initial state")
                
                # Save images
                save_tensor_images(condition, size=(input_dim, target_shape, target_shape), name=f"{epoch}_condition_{cur_step}.png")
                save_tensor_images(real, size=(real_dim, target_shape, target_shape), name=f"{epoch}_real_{cur_step}.png")
                save_tensor_images(fake, size=(real_dim, target_shape, target_shape), name=f"{epoch}_fake_{cur_step}.png")
                
                # Reset mean losses
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                
                # Save current model checkpoint
                if save_model:
                    model_dict = {
                        'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict(),
                        'epoch': epoch,
                        'step': cur_step
                    }
                    torch.save(model_dict, f"pix2pix_smp_{cur_step}.pth")
            
            cur_step += 1
        
        # Calculate average losses for the epoch
        epoch_gen_loss = running_generator_loss / n_batches
        epoch_disc_loss = running_discriminator_loss / n_batches
        
        print(f"Epoch {epoch} completed. Average generator loss: {epoch_gen_loss:.4f}, Average discriminator loss: {epoch_disc_loss:.4f}")
        
        # Check for early stopping
        model_dict = {
            'gen': gen.state_dict(),
            'gen_opt': gen_opt.state_dict(),
            'disc': disc.state_dict(),
            'disc_opt': disc_opt.state_dict(),
            'epoch': epoch,
            'step': cur_step,
            'generator_loss': epoch_gen_loss,
            'discriminator_loss': epoch_disc_loss
        }
        
        is_best = early_stopping(epoch_gen_loss, model_dict)
        if is_best:
            print(f"Improved generator loss from {best_generator_loss:.4f} to {epoch_gen_loss:.4f}, saving model...")
            best_generator_loss = epoch_gen_loss
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} epochs.")
            print(f"Best model saved at: {early_stopping.checkpoint_path}")
            break
    
    # Ensure the best model is used
    if os.path.exists(early_stopping.checkpoint_path):
        print(f"Loading best model from {early_stopping.checkpoint_path}")
        best_model = torch.load(early_stopping.checkpoint_path)
        gen.load_state_dict(best_model['gen'])
        disc.load_state_dict(best_model['disc'])
        
        # Save final model
        torch.save({
            'gen': gen.state_dict(),
            'gen_opt': gen_opt.state_dict(),
            'disc': disc.state_dict(),
            'disc_opt': disc_opt.state_dict(),
            'best_loss': best_generator_loss
        }, "final_pix2pix_smp.pth")
        
        print("Final model saved as 'final_pix2pix_smp.pth'")

# Run the training loop
if __name__ == "__main__":
    train() 
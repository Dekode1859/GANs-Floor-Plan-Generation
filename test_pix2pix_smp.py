import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import argparse

# Utility functions - removed display functionality
def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), filename='image.png'):
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(12, 12))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test Pix2Pix GAN with Pretrained UNet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input_dir', type=str, default='floors', help='Input directory with test images')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory for generated images')
    parser.add_argument('--num_test_images', type=int, default=5, help='Number of test images to generate')
    args = parser.parse_args()
    
    # Parameters
    input_dim = 3
    real_dim = 3
    target_shape = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model
    gen = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=input_dim,
        classes=real_dim,
        activation="sigmoid"
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gen.load_state_dict(checkpoint['gen'])
    gen.eval()  # Set to evaluation mode
    
    # Load test data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.ImageFolder(args.input_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Generate and save results
    print(f"Generating {args.num_test_images} test images...")
    count = 0
    
    with torch.no_grad():
        for image, _ in dataloader:
            if count >= args.num_test_images:
                break
            
            # Process image
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            condition = nn.functional.interpolate(condition, size=(target_shape, target_shape), mode='bilinear')
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=(target_shape, target_shape), mode='bilinear')
            
            # Move to device
            condition = condition.to(device)
            real = real.to(device)
            
            # Generate fake image
            fake = gen(condition)
            
            # Print progress
            print(f"Processing test image {count+1}")
            
            # Save the images
            save_tensor_images(condition, size=(input_dim, target_shape, target_shape), 
                           filename=f"{args.output_dir}/test_{count+1}_condition.png")
            save_tensor_images(real, size=(real_dim, target_shape, target_shape), 
                           filename=f"{args.output_dir}/test_{count+1}_real.png")
            save_tensor_images(fake, size=(real_dim, target_shape, target_shape), 
                           filename=f"{args.output_dir}/test_{count+1}_generated.png")
            
            # Create a side-by-side comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            condition_img = condition.detach().cpu()[0].permute(1, 2, 0)
            plt.imshow(condition_img)
            plt.title("Input Condition")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            real_img = real.detach().cpu()[0].permute(1, 2, 0)
            plt.imshow(real_img)
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            fake_img = fake.detach().cpu()[0].permute(1, 2, 0)
            plt.imshow(fake_img)
            plt.title("Generated")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/test_{count+1}_comparison.png")
            plt.close()
            
            count += 1
    
    print(f"Generated {count} test images in {args.output_dir}")

if __name__ == "__main__":
    main() 
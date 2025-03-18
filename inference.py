import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
import os
import argparse

def load_image(image_path, target_size=256):
    """Load and preprocess a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Determine if this is a paired image (condition + ground truth) or just condition
    width, height = image.size
    is_paired = width > height * 1.5  # Heuristic to detect side-by-side images
    
    if is_paired:
        # Split the image into condition (left) and ground truth (right)
        condition = image.crop((0, 0, width // 2, height))
    else:
        # Use the entire image as condition
        condition = image
    
    # Resize to target size
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])
    
    condition_tensor = transform(condition).unsqueeze(0)  # Add batch dimension
    
    # If we have a paired image, also return the ground truth
    if is_paired:
        ground_truth = image.crop((width // 2, 0, width, height))
        ground_truth_tensor = transform(ground_truth).unsqueeze(0)
        return condition_tensor, ground_truth_tensor, is_paired
    
    return condition_tensor, None, is_paired

def run_inference(model, condition_tensor, device):
    """Run inference with the model"""
    model.eval()
    with torch.no_grad():
        condition_tensor = condition_tensor.to(device)
        generated = model(condition_tensor)
    return generated

def save_result(condition, generated, ground_truth=None, output_path="output.png"):
    """Save the inference result as an image"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # If we have ground truth, show comparison
    if ground_truth is not None:
        # Create a side-by-side comparison
        plt.figure(figsize=(15, 5))
        
        # Input condition
        plt.subplot(1, 3, 1)
        condition_img = condition.cpu()[0].permute(1, 2, 0)
        plt.imshow(condition_img)
        plt.title("Input Condition")
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        real_img = ground_truth.cpu()[0].permute(1, 2, 0)
        plt.imshow(real_img)
        plt.title("Ground Truth")
        plt.axis('off')
        
        # Generated output
        plt.subplot(1, 3, 3)
        fake_img = generated.cpu()[0].permute(1, 2, 0)
        plt.imshow(fake_img)
        plt.title("Generated")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    else:
        # Just save condition and generated images
        plt.figure(figsize=(10, 5))
        
        # Input condition
        plt.subplot(1, 2, 1)
        condition_img = condition.cpu()[0].permute(1, 2, 0)
        plt.imshow(condition_img)
        plt.title("Input Condition")
        plt.axis('off')
        
        # Generated output
        plt.subplot(1, 2, 2)
        fake_img = generated.cpu()[0].permute(1, 2, 0)
        plt.imshow(fake_img)
        plt.title("Generated")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    # Also save just the generated image separately
    generated_output_path = output_path.replace('.png', '_generated.png')
    save_image(generated.cpu(), generated_output_path, normalize=True)
    
    print(f"Results saved to {output_path} and {generated_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with Pix2Pix model on a single image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_path', type=str, default='output.png', help='Path to save the output image')
    parser.add_argument('--target_size', type=int, default=256, help='Target size for input/output images')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    input_dim = 3
    real_dim = 3
    
    gen = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=input_dim,
        classes=real_dim,
        activation="sigmoid"
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'gen' in checkpoint:
        gen.load_state_dict(checkpoint['gen'])
        print(f"Loaded generator from checkpoint: {args.checkpoint}")
    else:
        # Try loading the model directly (in case it's not a GAN checkpoint)
        try:
            gen.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint: {args.checkpoint}")
        except:
            print(f"Could not load model from checkpoint: {args.checkpoint}")
            raise
    
    # Load and preprocess image
    condition_tensor, ground_truth_tensor, is_paired = load_image(args.image_path, args.target_size)
    
    # Run inference
    generated = run_inference(gen, condition_tensor, device)
    
    # Save result
    save_result(condition_tensor, generated, ground_truth_tensor, args.output_path)
    
    if is_paired:
        print("Processed paired image (condition + ground truth)")
    else:
        print("Processed single image (condition only)")

if __name__ == "__main__":
    main() 
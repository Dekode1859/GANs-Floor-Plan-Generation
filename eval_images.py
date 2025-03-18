import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
import os
import argparse
from tqdm import tqdm
import datetime

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'gen' in checkpoint:
        gen.load_state_dict(checkpoint['gen'])
        print(f"Loaded generator from checkpoint: {checkpoint_path}")
    else:
        # Try loading the model directly
        try:
            gen.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint: {checkpoint_path}")
        except:
            print(f"Could not load model from checkpoint: {checkpoint_path}")
            raise
    
    return gen

def load_and_process_image(image_path, target_size=256):
    """Load, split and preprocess a validation image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Split the image into condition (left) and ground truth (right)
    width, height = image.size
    condition = image.crop((0, 0, width // 2, height))
    ground_truth = image.crop((width // 2, 0, width, height))
    
    # Resize and convert to tensors
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])
    
    condition_tensor = transform(condition).unsqueeze(0)  # Add batch dimension
    ground_truth_tensor = transform(ground_truth).unsqueeze(0)  # Add batch dimension
    
    return condition_tensor, ground_truth_tensor

def run_inference(model, condition_tensor, device):
    """Run inference with the model"""
    model.eval()
    with torch.no_grad():
        condition_tensor = condition_tensor.to(device)
        generated = model(condition_tensor)
    return generated

def save_comparison(condition, real, generated, output_path, image_name=""):
    """Save a side-by-side comparison of condition, real, and generated images"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create a side-by-side comparison
    plt.figure(figsize=(15, 5))
    
    # Input condition
    plt.subplot(1, 3, 1)
    condition_img = condition.cpu()[0].permute(1, 2, 0)
    plt.imshow(condition_img)
    plt.title("Condition")
    plt.axis('off')
    
    # Ground truth
    plt.subplot(1, 3, 2)
    real_img = real.cpu()[0].permute(1, 2, 0)
    plt.imshow(real_img)
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Generated output
    plt.subplot(1, 3, 3)
    fake_img = generated.cpu()[0].permute(1, 2, 0)
    plt.imshow(fake_img)
    plt.title("Generated")
    plt.axis('off')
    
    # Add super title if image name is provided
    if image_name:
        plt.suptitle(image_name, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also save just the generated image separately
    generated_output_path = output_path.replace('.png', '_generated.png')
    save_image(generated.cpu(), generated_output_path, normalize=True)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pix2Pix model on validation images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--val_dir', type=str, default='floors/val', help='Directory containing validation images')
    parser.add_argument('--output_dir', type=str, default='val_results', help='Directory to save evaluation results')
    parser.add_argument('--target_size', type=int, default=256, help='Target size for input/output images')
    parser.add_argument('--image_ext', type=str, default='.png', help='Image file extension to process')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get validation images
    val_images = [f for f in os.listdir(args.val_dir) if f.endswith(args.image_ext)]
    
    if not val_images:
        print(f"No validation images found in {args.val_dir} with extension {args.image_ext}")
        return
    
    print(f"Found {len(val_images)} validation images")
    
    # Create a grid output figure
    n_cols = 3  # condition, real, generated
    n_rows = min(len(val_images), 5)  # Show at most 5 examples in the grid
    
    grid_fig = plt.figure(figsize=(15, n_rows * 4))
    
    # Process each validation image
    for i, image_file in enumerate(tqdm(val_images, desc="Processing validation images")):
        image_path = os.path.join(args.val_dir, image_file)
        
        # Load and process image
        condition_tensor, ground_truth_tensor = load_and_process_image(image_path, args.target_size)
        
        # Run inference
        generated_tensor = run_inference(model, condition_tensor, device)
        
        # Create output path for individual comparison
        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
        
        # Save individual comparison
        save_comparison(condition_tensor, ground_truth_tensor, generated_tensor, output_path, image_name=base_name)
        
        # Add to grid figure (first 5 images only)
        if i < n_rows:
            # Condition
            plt.subplot(n_rows, n_cols, i*n_cols + 1)
            condition_img = condition_tensor.cpu()[0].permute(1, 2, 0)
            plt.imshow(condition_img)
            if i == 0:
                plt.title("Condition")
            plt.axis('off')
            
            # Ground truth
            plt.subplot(n_rows, n_cols, i*n_cols + 2)
            real_img = ground_truth_tensor.cpu()[0].permute(1, 2, 0)
            plt.imshow(real_img)
            if i == 0:
                plt.title("Ground Truth")
            plt.axis('off')
            
            # Generated
            plt.subplot(n_rows, n_cols, i*n_cols + 3)
            fake_img = generated_tensor.cpu()[0].permute(1, 2, 0)
            plt.imshow(fake_img)
            if i == 0:
                plt.title("Generated")
            plt.axis('off')
            
            # Add image name as y-axis label for the row
            plt.ylabel(base_name, rotation=0, labelpad=50, ha='right')
    
    # Save grid figure
    plt.tight_layout()
    grid_output_path = os.path.join(args.output_dir, "all_comparisons_grid.png")
    plt.savefig(grid_output_path, bbox_inches='tight')
    plt.close()
    
    # Create a combined comparison of all results
    # Create an HTML report for easy viewing
    html_path = os.path.join(args.output_dir, "results.html")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Floor Plan GAN Validation Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2 {{
                    color: #333;
                }}
                .result-grid {{
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .comparison {{
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .comparison img {{
                    width: 100%;
                    height: auto;
                }}
                .comparison h3 {{
                    text-align: center;
                    margin-bottom: 10px;
                }}
                .summary {{
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <h1>Floor Plan GAN Validation Results</h1>
            <div class="summary">
                <p>Model checkpoint: {args.checkpoint}</p>
                <p>Total validation images: {len(val_images)}</p>
                <p>Generated on: {current_time}</p>
            </div>
            
            <h2>Overview</h2>
            <div class="comparison">
                <img src="all_comparisons_grid.png" alt="All Results Grid">
            </div>
            
            <h2>Individual Results</h2>
            <div class="result-grid">
        """)
        
        # Add each individual comparison
        for image_file in val_images:
            base_name = os.path.splitext(image_file)[0]
            f.write(f"""
                <div class="comparison">
                    <h3>{base_name}</h3>
                    <img src="{base_name}_comparison.png" alt="{base_name} comparison">
                </div>
            """)
        
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"HTML report generated at {html_path}")

if __name__ == "__main__":
    main()

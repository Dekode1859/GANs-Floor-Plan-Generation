# Floor Plan GAN using Pix2Pix

This repository contains two implementations of Pix2Pix GAN for floor plan image-to-image translation:

1. **Original Pix2Pix Implementation** - A custom UNet generator and discriminator built from scratch
2. **Pretrained UNet Pix2Pix Implementation** - Using a pretrained UNet from the Segmentation Models PyTorch (SMP) library

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The data should be organized in the following structure:

```
floors/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each image should contain both the input (condition) and target side by side. The left half of the image will be used as the condition, and the right half as the ground truth.

## Original Pix2Pix Implementation

### Training

Run the original Pix2Pix model:

```bash
python Pix2Pix_GAN.ipynb
```

Or if you prefer to run it as a script:

```bash
jupyter nbconvert --to python Pix2Pix_GAN.ipynb
python Pix2Pix_GAN.py
```

### Using a Pretrained Model

To use a pretrained model from the original implementation:

```python
# Load the model
loaded_state = torch.load("pix2pix_15000.pth")
gen.load_state_dict(loaded_state["gen"])
gen_opt.load_state_dict(loaded_state["gen_opt"])
disc.load_state_dict(loaded_state["disc"])
disc_opt.load_state_dict(loaded_state["disc_opt"])
```

## SMP Pretrained UNet Pix2Pix Implementation

This implementation uses a pretrained UNet from the Segmentation Models PyTorch library as the generator.

### Training

To train the SMP Pix2Pix model:

```bash
python pix2pix_smp.py
```

The model will save checkpoints during training in the format `pix2pix_smp_{step}.pth`. Additionally, it uses early stopping to monitor the generator loss and save the best model when improvement is detected. If no improvement is seen for a specified number of epochs, training stops and the best model is loaded.

Key early stopping parameters (defined in the script):
- `patience`: Number of epochs with no improvement after which training will stop (default: 10)
- `min_delta`: Minimum change in loss to qualify as an improvement (default: 0.001)

The best model will be saved as `best_pix2pix_smp.pth` and the final model (after loading the best weights) as `final_pix2pix_smp.pth`.

### Testing and Inference

After training, you can test the model on new images:

```bash
python test_pix2pix_smp.py --checkpoint final_pix2pix_smp.pth --input_dir floors --output_dir test_results --num_test_images 5
```

Arguments:
- `--checkpoint`: Path to the trained model checkpoint
- `--input_dir`: Directory containing test images
- `--output_dir`: Directory to save generated outputs
- `--num_test_images`: Number of test images to generate

## Differences Between Implementations

1. **Generator Architecture**:
   - Original: Custom UNet implementation with multiple contracting and expanding blocks
   - SMP Version: Pretrained ResNet34 encoder with UNet decoder architecture

2. **Pretrained Weights**:
   - Original: Randomly initialized
   - SMP Version: ImageNet pretrained weights for better feature extraction

3. **Performance**:
   - The SMP version typically converges faster and produces better results due to transfer learning from pretrained weights

4. **Customization**:
   - The SMP version allows easy swapping of different encoder architectures (ResNet, EfficientNet, etc.)

5. **Early Stopping**:
   - The SMP version implements early stopping to prevent overfitting and save the best model automatically

## Tips for Optimal Results

1. **Data Preparation**:
   - Make sure images are properly aligned (condition on left, ground truth on right)
   - Use consistent image sizes

2. **Training Parameters**:
   - Adjust `lambda_recon` to control the balance between adversarial and reconstruction loss
   - Modify learning rate for different convergence behavior
   - Tune early stopping parameters (`patience` and `min_delta`) based on your dataset size and complexity

3. **Model Selection**:
   - For SMP version, try different encoder architectures:
     ```python
     gen = smp.Unet(
         encoder_name="efficientnet-b0",  # Alternative: "resnet50", "mobilenet_v2", etc.
         encoder_weights="imagenet",
         in_channels=input_dim,
         classes=real_dim,
         activation="sigmoid"
     )
     ```

4. **For Large Datasets**:
   - Increase batch size if memory allows
   - Use a more powerful encoder for the SMP version

## Output Visualization

The implementation saves all images to disk without displaying them during training/testing:

- During training, images are saved to the `floor_runs/` directory, including:
  - Condition images (inputs)
  - Real images (ground truth)
  - Generated images (outputs)

- During testing, the following are saved to the specified output directory:
  - Individual condition, real, and generated images
  - Side-by-side comparisons to help evaluate the model's performance 
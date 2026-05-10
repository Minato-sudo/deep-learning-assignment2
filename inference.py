import os
import torch
from diffusers import DDPMPipeline

def run_inference():
    print("--- Running Basic Inference ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/ddpm-cifar10-32"
    
    print(f"Loading pretrained model: {model_id}")
    pipeline = DDPMPipeline.from_pretrained(model_id).to(device)
    
    print("Generating 1 sample image...")
    # Generate a single image
    with torch.no_grad():
        image = pipeline(batch_size=1, num_inference_steps=50).images[0]
        
    os.makedirs("results", exist_ok=True)
    save_path = "results/sample_inference_output.png"
    image.save(save_path)
    
    print(f"Success! Image saved to {save_path}")
    print("For full interactive inference and experiments, please run notebooks/01_inference_demo.ipynb")

if __name__ == "__main__":
    run_inference()

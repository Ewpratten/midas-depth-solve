import argparse
import os
import cv2
import torch
import sys


def main() -> int:
    # Handle program arguments
    ap = argparse.ArgumentParser(
        prog='solve.py', description='solves 3D depth from 2D images')
    ap.add_argument('input_dir',  help='Directory containing input images')
    ap.add_argument('output_dir', help='Directory to write output images')
    args = ap.parse_args()

    # Allow configuration changes through the environment
    midas_model = os.environ.get('MIDAS_MODEL', 'DPT_Large')
    force_use_cpu = "FORCE_CPU_COMPUTE" in os.environ

    # Make sure both input and output directories exist
    if not os.path.isdir(args.input_dir):
        print('Input directory does not exist')
        return 1
    if not os.path.isdir(args.output_dir):
        print('Output directory does not exist')
        return 1

    # Search for all images in input directory
    input_files = os.listdir(args.input_dir)
    print(f"Found {len(input_files)} images in input directory:")
    for file in input_files:
        print(f"\t- {file}")

    # Load the model
    print(f"Loading model: {midas_model}")
    midas = torch.hub.load("intel-isl/midas", midas_model)

    # If possible, use the GPU
    device = None
    if torch.cuda.is_available():
        print("CUDA support is available")
        if not force_use_cpu:
            print("Using GPU for computation")
            device = torch.device("cuda")
        else:
            print("$FORCE_CPU_COMPUTE is set. Overriding compute settings to use CPU")
            device = torch.device("cpu")
    else:
        print("CUDA support is not available. Using CPU")
        device = torch.device("cpu")

    # Move the model to the device
    midas.to(device)
    midas.eval()

    # Process all images
    for file in input_files:
        print(f"Processing image: {file}")

        # Load the transforms to normalize the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if midas_model == "DPT_Large" or midas_model == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Load the image and transform it
        img = cv2.imread(os.path.join(args.input_dir, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        # Compute the depth of the image
        print("Computing image depth... Please wait")
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        print("Depth computation complete!")

        # Save the output image
        output_file = os.path.join(args.output_dir, file)
        cv2.imwrite(output_file, output)
        print(f"Saved output image to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

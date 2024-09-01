import sys
import os
from PIL import Image, ImageFilter


def apply_blur(input_dir, output_dir, sigma_value):
    """
    Apply Gaussian blur to all images in the input directory and save them to the output directory.

    :param input_dir: Path to the directory containing input images.
    :param output_dir: Path to the directory where blurred images will be saved.
    :param sigma_value: Sigma value for Gaussian blur.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open the image file
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Apply Gaussian blur
            blurred_img = img.filter(ImageFilter.GaussianBlur(sigma_value))

            # Save the blurred image to the output directory
            output_path = os.path.join(output_dir, filename)
            blurred_img.save(output_path)

            print(f'Processed {filename}')

    print('All images have been processed.')


def compress(input_dir, output_dir):
    """
    Compress images using JPEG at quality levels of 75, then resize images by down-sampling
    to half of their original size and then up-sampling back using nearest-neighbor interpolation.

    :param input_dir: Path to the directory containing input images.
    :param output_dir: Path to the directory where processed images will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open the image file
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Compress the image using JPEG with quality level of 75
            compressed_path = os.path.join(output_dir, f"compressed_{filename}")
            img.save(compressed_path, format="JPEG", quality=75)

            # Open the compressed image
            compressed_img = Image.open(compressed_path)

            # Resize the image to half its size
            half_size = (compressed_img.size[0] // 2, compressed_img.size[1] // 2)
            img_resized = compressed_img.resize(half_size, Image.NEAREST)

            # Resize the image back to its original size
            img_resized_back = img_resized.resize(compressed_img.size, Image.NEAREST)

            # Save the resized image to the output directory
            final_output_path = os.path.join(output_dir, filename)
            img_resized_back.save(final_output_path, format="JPEG", quality=75)

            print(f'Compressed and resized {filename}')

    print('All images have been compressed and resized.')


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python script_name.py <function_name> <input_directory> <output_directory> <sigma_value_or_empty>")
        return

    function_name = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    if function_name == "apply_blur":
        sigma = float(sys.argv[4])
        apply_blur(input_directory, output_directory, sigma)
    elif function_name == "compress":
        compress(input_directory, output_directory)
    else:
        print(f"Function '{function_name}' not recognized.")


if __name__ == "__main__":
    main()

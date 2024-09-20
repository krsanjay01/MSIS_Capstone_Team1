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


def resize(input_dir, output_dir):
    """
    Resize images by down-sampling
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

            # Resize the image to half its size
            half_size = (img.size[0] // 2, img.size[1] // 2)
            img_resized = img.resize(half_size, Image.NEAREST)

            # Resize the image back to its original size
            img_resized_back = img_resized.resize(img.size, Image.NEAREST)

            # Save the resized image to the output directory
            final_output_path = os.path.join(output_dir, filename)
            img_resized_back.save(final_output_path, format="JPEG", quality=75)

    print('All images have been resized.')

def compress_jpeg(input_dir, output_dir, compression_level):
    """
    Compress images using JPEG at quality levels of 75

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

            # If the image has an alpha channel (RGBA), convert it to RGB
            if img.mode == 'RGB':

                # Get the filename without extension and add .jpg extension
                base_name = os.path.splitext(filename)[0]
                compressed_path = os.path.join(output_dir, f"{base_name}.jpg")

                print('Writing image: ', compressed_path, '...')
                # Compress the image using JPEG with quality level of 75
                img.save(compressed_path, format="JPEG", quality=compression_level)

    print('All images have been compressed to JPEG 75.')

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
        compression_level = int(sys.argv[4])
        compress_jpeg(input_directory, output_directory, compression_level)
    else:
        print(f"Function '{function_name}' not recognized.")


if __name__ == "__main__":
    main()

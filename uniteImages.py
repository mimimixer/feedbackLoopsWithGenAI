from PIL import Image
import os

# Example usage:
folder_name = ("generatedT2I_50-parallelLines")
folder_path = f"./genT2Image/{folder_name}"  # Replace with your folder path
output_path = f"./{folder_path}/50.concatenated_parallelLines.jpg"  # Replace with your desired output path

def concatenate_images(folder_path, output_path):
    # Get all image file names in the folder (sorted to maintain order)
    image_files = sorted([file for file in os.listdir(folder_path) if file.endswith(('png', 'PNG', 'jpg', 'jpeg'))])
    print(image_files)

    # Open all images and get their dimensions
    images = [Image.open(os.path.join(folder_path, file)) for file in image_files]

    # Ensure all images have the same height by resizing (optional, remove if not needed)
    #max_height = max(img.height for img in images)
    max_height = 512
    #images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in images]

    # Calculate the total width of the concatenated image
    total_width = 512*len(images) #sum(img.width for img in images)

    # Create a blank canvas for the final image
    concatenated_image = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    current_x = 0
    for img in images:
        concatenated_image.paste(img, (current_x, 0))
        current_x += 512

    # Save the final concatenated image
    concatenated_image.save(output_path)
    print(f"Concatenated image saved at {output_path}")

concatenate_images(folder_path, output_path)
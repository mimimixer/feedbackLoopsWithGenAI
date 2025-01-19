from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the image2text model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the text2image model and processor
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# Load the image
first_image = Image.open("./genImage/generated28-noise/generated28-noise_005.jpg")
plt.imshow(first_image)
plt.axis('off')
plt.show()
generated_image = first_image

# define naming
name_of_current_run = ("generated28-noise")
output_directory = f"./genImage/{name_of_current_run}/"
os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

def generate_text_on_image(input):
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    print("Generated Caption:", caption)
    caption_path = f"./genImage/{name_of_current_run}/{name_of_current_run}.txt"

    # Read the existing content of the file
    try:
        with open(caption_path, "r") as f:
            existing_content = f.read()
    except FileNotFoundError:
        # If the file doesn't exist, start with empty content
        existing_content = ""

    # Add the new caption as the first line, numbered
    new_content = f"{i}. {caption}\n{existing_content}"
    # Add the new caption with the correct line number
    #new_content = f"{i}. {caption}\n{''.join(existing_lines)}"

    # Write the new content back to the file
    with open(caption_path, "w") as f:
        f.write(new_content)
    return caption

def generate_image_on_text(prompt):
    generatedImage = pipe(prompt).images[0]
    #processed_image = np.nan_to_num(generatedImage)  # Replace NaN values with 0
    generatedImage_path =  f"./genImage/{name_of_current_run}/{name_of_current_run}_{i+1:03}.jpg"
    generatedImage.save(generatedImage_path)
    generatedImage.show()
    plt.imshow(generatedImage)
    plt.axis('off')  # Optional: Hide axes
    plt.show()
    return generatedImage

for i in range (6,11):
    # Generate text from image
    if i==1:
        inputs = processor(images=[first_image], return_tensors="pt")
        #first_image.save(f"./genImage/{name_of_current_run}/{name_of_current_run}_startImage_{first_image}.jpg")
    else:
        inputs = processor(images=[generated_image], return_tensors="pt")
    generated_caption = generate_text_on_image(inputs)

    # Generate image from text
    #prompt = "An image of a squirrel in Picasso style"
    # generatedImage = pipe(prompt,  height=height, width=width).images[0]
    prompt = generated_caption
    generated_image= generate_image_on_text(prompt)
i=5
generate_text_on_image(generated_image)
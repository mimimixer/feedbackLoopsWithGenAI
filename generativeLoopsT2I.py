from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the text2image model and processor
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# Load the image2text model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# define naming
name_of_current_run = ("generatedT2I_50-parallelLines")
output_directory = f"./genT2Image/{name_of_current_run}/"
os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

# Load the prompt
first_prompt = "15 parallel black lines"
generated_caption = first_prompt
print("First prompt:", first_prompt)
caption_path = f"./genT2Image/{name_of_current_run}/{name_of_current_run}.txt"
# Read the existing content of the file


def writeTextIntoFile(caption_path, number, text):
    try:
        with open(caption_path, "r") as f:
            existing_content = f.read()
    except FileNotFoundError:
        # If the file doesn't exist, start with empty content
        existing_content = ""
    # Add the new caption as the first line, numbered
    new_content = f"{number}. {text}\n{existing_content}"
    # Write the new content back to the file
    with open(caption_path, "w") as f:
        f.write(new_content)
    return text

def generate_text_on_image(input):
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    print("Generated Caption:", caption)
    caption_path = f"./genT2Image/{name_of_current_run}/{name_of_current_run}.txt"
    #writeTextIntoFile(caption_path, i+1, caption)

    # Read the existing content of the file
    try:
        with open(caption_path, "r") as f:
            existing_content = f.read()
    except FileNotFoundError:
        # If the file doesn't exist, start with empty content
        existing_content = ""

    # Add the new caption as the first line, numbered
    new_content = f"{i+1}. {caption}\n{existing_content}"
    # Add the new caption with the correct line number
    #new_content = f"{i}. {caption}\n{''.join(existing_lines)}"

    # Write the new content back to the file
    with open(caption_path, "w") as f:
        f.write(new_content)
    return caption

def generate_image_on_text(prompt):
    generatedImage = pipe(prompt).images[0]
    #processed_image = np.nan_to_num(generatedImage)  # Replace NaN values with 0
    generatedImage_path =  f"./genT2Image/{name_of_current_run}/{name_of_current_run}_I{i+1:03}.jpg"
    generatedImage.save(generatedImage_path)
    generatedImage.show()
    plt.imshow(generatedImage)
    plt.axis('off')  # Optional: Hide axes
    plt.show()
    return generatedImage


writeTextIntoFile(caption_path, 0, first_prompt)
for i in range (0,5):
    # Generate image from text
    prompt = generated_caption
    generated_image = generate_image_on_text(prompt)

    #generate text from image
    inputs = processor(images=[generated_image], return_tensors="pt")
    generated_caption = generate_text_on_image(inputs)

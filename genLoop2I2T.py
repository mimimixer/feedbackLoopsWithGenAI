from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, GitVisionModel
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import uniteImages

# Load the image2text model and processor
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#genai.configure(api_key="AIzaSyDTfRFeL-I6DO2NzvDkl_RIhvuKFV1ABYs")
#model = genai.GenerativeModel("gemini-1.5-flash")
#processor = AutoProcessor.from_pretrained("microsoft/git-base")
#model = GitVisionModel.from_pretrained("microsoft/git-base")
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")


# Load the text2image model and processor
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("mps")
# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# Load the image
image_name = "./dasGelbeQuadrat.jpg"
first_image = Image.open(image_name)
plt.imshow(first_image)
plt.axis('off')
plt.show()
generated_image = first_image

# define naming
current_run = os.path.splitext(os.path.basename(image_name))[0]
print(current_run)
number_of_current_run = "12"
name_of_current_run = (f"generated{number_of_current_run}-{current_run}")
output_directory = f"./gen2Image2T/{name_of_current_run}/"
os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist
first_image.save(f"./gen2Image2T/{name_of_current_run}/{name_of_current_run}_000_startImage.jpg")
concaternated_image_name = f"{number_of_current_run}_concatenated-{current_run}.jpg"

def generate_text_on_image(input_image):
    #output = model.generate(**inputs)
    #caption = processor.decode(output[0], skip_special_tokens=True)
    pixel_values = processor(images=input_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Generated Caption:", generated_caption)
    caption_path = f"./gen2Image2T/{name_of_current_run}/{name_of_current_run}.txt"

    # Read the existing content of the file
    try:
        with open(caption_path, "r") as f:
            existing_content = f.read()
    except FileNotFoundError:
        # If the file doesn't exist, start with empty content
        existing_content = ""

    # Add the new caption as the first line, numbered
    if (existing_content == ""):
        new_content = f"{i}.{generated_caption}"
    else:
        new_content = f"{existing_content}\n{i}.{generated_caption}"
    # Add the new caption with the correct line number
    #new_content = f"{i}. {caption}\n{''.join(existing_lines)}"

    # Write the new content back to the file
    with open(caption_path, "w") as f:
        f.write(new_content)
    return generated_caption

def generate_image_on_text(prompt):
    generatedImage = pipe(prompt).images[0]
    #processed_image = np.nan_to_num(generatedImage)  # Replace NaN values with 0
    generatedImage_path =  f"./gen2Image2T/{name_of_current_run}/{name_of_current_run}_{i+1:03}.jpg"
    generatedImage.save(generatedImage_path)
    generatedImage.show()
    plt.imshow(generatedImage)
    plt.axis('off')  # Optional: Hide axes
    plt.show()
    return generatedImage

for i in range (0,5):
    generated_caption = generate_text_on_image(generated_image)

    # Generate image from text
    #prompt = "An image of a squirrel in Picasso style"
    # generatedImage = pipe(prompt,  height=height, width=width).images[0]
    prompt = generated_caption
    generated_image= generate_image_on_text(prompt)
i=5
generate_text_on_image(generated_image)

uniteImages.concatenate_images(output_directory, name_of_current_run)
from torch.utils.data import DataLoader
from PIL import Image
import os
from auto_image_captioning import inference
import json
from tqdm import tqdm
import datasets


dataset = datasets.load_dataset("ossaili/simple_arch_1400")

if not os.path.exists("images"):
    os.mkdir("images")

# Wrap the dataset in tqdm to display a progress bar
for idx, sample in tqdm(enumerate(dataset['train'])):
    image = sample['image']
    # image_pil = Image.fromarray(image)
    text = inference(image, 'Image Captioning', '', '')
    filename = f"{idx}.jpg"
    filepath = os.path.join("images", filename)

    # Open the file in append mode
    with open(os.path.join("images", 'metadata.jsonl'), 'a') as f:
        # Create a dictionary with some metadata
        metadata = {'filename': filename, 'text': text}
        # Convert the metadata to a JSON string
        metadata_json = json.dumps(metadata)
        # Write the JSON string to the file followed by a newline character
        f.write(metadata_json + '\n')

    image.save(filepath)

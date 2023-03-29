import os
import json
from tqdm import tqdm
import argparse
import datasets
from torch.utils.data import DataLoader
from PIL import Image
from blip_inference import load_models, caption_image


def process_images(output_dir, dataset_name):
    dataset = datasets.load_dataset(dataset_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Load the models and transforms
    model, transform = load_models()

    # Wrap the dataset in tqdm to display a progress bar
    for idx, sample in tqdm(enumerate(dataset['train'])):
        image = Image.fromarray(sample['image'])
        text = caption_image(image, model, transform)
        filename = f"{idx}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Open the file in append mode
        with open(os.path.join(output_dir, 'metadata.jsonl'), 'a') as f:
            # Create a dictionary with some metadata
            metadata = {'file_name': filename, 'text': text}
            # Convert the metadata to a JSON string
            metadata_json = json.dumps(metadata)
            # Write the JSON string to the file followed by a newline character
            f.write(metadata_json + '\n')

        image.save(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Process images and generate captions")
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Directory to store the processed images and metadata")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="The name of the dataset to process")
    args = parser.parse_args()

    process_images(args.output_dir, args.dataset_name)


if __name__ == '__main__':
    main()

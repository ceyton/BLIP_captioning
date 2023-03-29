import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_vqa import blip_vqa
from models.blip import blip_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    # Load the Image Captioning model
    image_size = 384
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
    model.eval()
    model = model.to(device)

    return model, transform


def caption_image(raw_image, model=None, transform=None):
    if model is None or transform is None:
        model, transform = load_models()

    image = transform(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        caption = model.generate(
            image, sample=False, num_beams=3, max_length=20, min_length=5)
    return caption[0]

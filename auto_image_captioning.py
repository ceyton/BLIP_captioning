from models.blip_vqa import blip_vqa
from models.blip import blip_decoder
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


image_size_vq = 480
transform_vq = transforms.Compose([
    transforms.Resize((image_size_vq, image_size_vq),
                      interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'

model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
model_vq.eval()
model_vq = model_vq.to(device)


def inference(raw_image, model_n, question, strategy):
    if model_n == 'Image Captioning':
        image = transform(raw_image).unsqueeze(0).to(device)
        with torch.no_grad():
            if strategy == "Beam search":
                caption = model.generate(
                    image, sample=False, num_beams=3, max_length=20, min_length=5)
            else:
                caption = model.generate(
                    image, sample=True, top_p=0.9, max_length=20, min_length=5)
            return caption[0]

    else:
        image_vq = transform_vq(raw_image).unsqueeze(0).to(device)
        with torch.no_grad():
            answer = model_vq(image_vq, question,
                              train=False, inference='generate')
        return 'answer: '+answer[0]


# # Set the path to your image folder
# folder_path = "../archdaily_crwaler/output"

# # Loop through each file in the folder
# for filename in os.listdir(folder_path):
#     # Check if the file is an image file
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         # Open the image file
#         image_path = os.path.join(folder_path, filename)
#         image = Image.open(image_path)
#         print(inference(image, 'Image Captioning', '', ''))

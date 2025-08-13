import torch
import torchvision.transforms as T
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINOv2 model from timm
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True).to(device)
model.eval()

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])


def compute_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.forward_features(image_tensor)
        return features.squeeze().cpu().numpy()

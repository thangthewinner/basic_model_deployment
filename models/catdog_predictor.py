import sys 
import torch 
import torchvision 

from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image 
from torch.nn import functional as F
from utils.logger import Logger 
from configs.catdog_cfg import CatDogDataConfig 
from .catdog_model import CatDogModel 
import torchvision.transforms as transforms

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info('Starting Model Serving')

class Predictor:
    def __init__(self, model_name: str, model_weight: str, device: str = 'cpu'):
        self.model_name = model_name 
        self.model_weight = model_weight 
        self.device = device 
        self.load_model(n_classes=CatDogDataConfig.N_CLASSES)
        self.create_transform(
            img_size=CatDogDataConfig.IMG_SIZE,
            normalize_mean=CatDogDataConfig.NORMALIZE_MEAN,
            normalize_std=CatDogDataConfig.NORMALIZE_STD
        )

    def load_model(self, n_classes: int):
        self.model = CatDogModel(n_classes)

        map_location = torch.device(self.device)

        self.model.load_state_dict(torch.load(self.model_weight, map_location=map_location))
        self.model.to(map_location)
        self.model.eval()

    def create_transform(self, img_size: int, normalize_mean: list, normalize_std: list):
        self.transforms_ = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  
            transforms.Normalize(mean=normalize_mean, std=normalize_std) 
        ])
    
    def model_inference(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
        return output

    def output2pred(self, output):
        probs = F.softmax(output, dim=1).cpu().numpy()
        best_prob = probs.max()
        predicted_id = probs.argmax()
        predicted_class = "Cat" if predicted_id == 0 else "Dog"
        return probs.tolist(), best_prob, int(predicted_id), predicted_class

    async def predict(self, image):
        pil_img = Image.open(image) 

        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        transformed_image = self.transforms_(pil_img).unsqueeze(0)
        output = self.model_inference(transformed_image) 
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output) 

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class) 

        torch.cuda.empty_cache()

        resp_dict = {
            'probs': probs,
            'best_prob': best_prob, 
            'predicted_id': predicted_id,
            'predicted_class': predicted_class,
            'predictor_name': self.model_name
        }

        return resp_dict
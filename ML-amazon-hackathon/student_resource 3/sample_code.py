import os
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import src.constants
from tqdm import tqdm

class EntityValuePredictor(nn.Module):
    def __init__(self, num_entity_types):
        super(EntityValuePredictor, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, num_entity_types)
        self.value_head = nn.Linear(512, 1)  # Predict a single value
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.fc1(x))
        entity_type_logits = self.fc2(x)
        value = self.value_head(x)
        return entity_type_logits, value

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def load_model(model_path):
    entity_types = list(src.constants.entity_unit_map.keys())
    num_entity_types = len(entity_types)
    model = EntityValuePredictor(num_entity_types)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, entity_types

def predict(model, image_tensor, entity_name, entity_types):
    with torch.no_grad():
        entity_type_logits, value = model(image_tensor.unsqueeze(0))
    
    entity_type_probs = torch.softmax(entity_type_logits, dim=1)
    predicted_entity_index = torch.argmax(entity_type_probs).item()
    predicted_entity = entity_types[predicted_entity_index]
    
    if predicted_entity != entity_name:
        return ""  # Return empty string if predicted entity doesn't match
    
    predicted_value = value.item()
    
    # Choose appropriate unit based on entity type
    unit = next(iter(src.constants.entity_unit_map[entity_name]))
    
    # Format the prediction string
    return f"{predicted_value:.2f} {unit}"

def predictor(image_link, group_id, entity_name, model, entity_types):
    try:
        image = download_image(image_link)
        image_tensor = preprocess(image)
        prediction = predict(model, image_tensor, entity_name, entity_types)
        return prediction
    except Exception as e:
        print(f"Error processing {image_link}: {str(e)}")
        return ""

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    MODEL_PATH = 'entity_value_predictor.pth'  # Path to your trained model
    
    # Load the trained model
    model, entity_types = load_model(MODEL_PATH)
    
    # Load test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Make predictions
    tqdm.pandas()
    test['prediction'] = test.progress_apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model, entity_types), axis=1)
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)

    print(f"Output file generated: {output_filename}")
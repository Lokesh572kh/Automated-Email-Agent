import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

import torch
from transformers import RobertaTokenizer, RobertaModel

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)  

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # Hidden state from the last layer
        pooler = hidden_state[:, 0]  # Take the [CLS] token's hidden state
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class_labels = {0: 'student', 1: 'researcher', 2: 'corporates'}

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "src/pytorch_roberta_email_state_dict.pt"

model = None

def load_model():
    global model
    if model is None:
        # Instantiate the model architecture
        model_instance = RobertaClass()
        # Load the state dictionary
        model_instance.load_state_dict(torch.load(model_path, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        model = model_instance

def classify_email(email_text):
    load_model()
    inputs = tokenizer(email_text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    predicted_class_idx = predicted_class.item()
    predicted_class_label = class_labels.get(predicted_class_idx, 'Unknown')
    return predicted_class_label

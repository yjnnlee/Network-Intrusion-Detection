import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

dataset = pd.read_csv("E:/OneDrive/UNR-IDD.csv")

dataset.columns = dataset.columns.str.strip().astype(str)
dataset.rename(columns={'Port alive Duration (S)': 'PortAliveDuration', 
                   'Packets Matched': 'PacketsMatched'}, inplace=True)
    
df = dataset[['PortAliveDuration', 'PacketsMatched', 'Label']]

# Encoding
label_encoder = LabelEncoder()
df['Encoded Label'] = label_encoder.fit_transform(df['Label'])

# Training/Test Data Separation 8:2
X = df[['PortAliveDuration', 'PacketsMatched']].values
y = df['Encoded Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.inputs = tokenizer(
            [f"Port Duration: {x[0]}, Packets Matched: {x[1]}" for x in features],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# PyTorch Dataset
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# DistilBERT
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # number of epochs = 5 or 10 or ...
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()


from sklearn.metrics import accuracy_score, classification_report

predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

accuracy = accuracy_score(y_test, predicted_labels)
print(f"Test Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=label_encoder.classes_))

# Kibana
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# User infor
CLOUD_ID = 'cloud id'
ELASTIC_USERNAME = 'id'
ELASTIC_PASSWORD = 'pw'

es = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD)
)

results = pd.DataFrame({
    'PortAliveDuration': X_test[:, 0],
    'PacketsMatched': X_test[:, 1],
    'ActualLabel': label_encoder.inverse_transform(y_test),
    'PredictedLabel': label_encoder.inverse_transform(predicted_labels)
})

def generate_documents(df):
    for _, row in df.iterrows():
        yield {
            "_index": "ml_predictions",
            "_source": row.to_dict()
        }

# data upload to Kibana
bulk(es, generate_documents(results))
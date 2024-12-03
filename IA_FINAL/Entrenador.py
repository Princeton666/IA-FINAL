#
# CODIGO USADO PARA ENTRENAR EL MODELO, NO IMPORTANTE PARA LA COMPILACIÓN
#

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import classification_report

# Instalación requerida (si aún no se ha instalado)
# pip install pandas scikit-learn torch torchvision torchaudio transformers[torch] accelerate
#USANDO ESTE DATASET  ->    https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

ruta_carpeta = input("Por favor, introduce la ruta de la carpeta donde se encuentra el archivo 'sentiment140.csv': ")

ruta_archivo = f"{ruta_carpeta}/training.1600000.processed.noemoticon.csv"

dataset = pd.read_csv(ruta_archivo, encoding='latin-1', header=None)
dataset.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

dataset = dataset[['target', 'text']]

sentiment_mapping = {
    0: 0,  # Negativo
    2: 1,  # Neutral
    4: 2   # Positivo
}
dataset['target'] = dataset['target'].map(sentiment_mapping)

def limpiar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)  
    texto = re.sub(r"@\w+", "", texto)    
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)  
    texto = texto.lower().strip()  
    return texto

dataset['text'] = dataset['text'].fillna("").astype(str).apply(limpiar_texto)

X_train, X_test, y_train, y_test = train_test_split(
    dataset['text'], dataset['target'], test_size=0.2, random_state=42
)

y_train = y_train.reset_index(drop=True).astype(int)
y_test = y_test.reset_index(drop=True).astype(int)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="pt")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128, return_tensors="pt")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = Dataset(train_encodings, y_train.values)
test_dataset = Dataset(test_encodings, y_test.values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=7,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), axis=1)
print(classification_report(y_test, pred_labels, target_names=['Negativo', 'Neutral', 'Positivo']))

modelo.save_pretrained("./sentiment140_modelo")
tokenizer.save_pretrained("./sentiment140_modelo")
print("Entrenamiento y evaluación completados. Modelo guardado en './sentiment140_modelo'")

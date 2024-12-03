from flask import Flask, request, render_template
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import re

app = Flask(__name__)

# Ruta del modelo entrenado  // ACÁ DEBE CAMBIARLO
modelo_carpeta = "C:\\Users\\sergi\\OneDrive\\Documentos\\results\\results\\checkpoint-17500"

try:
    tokenizer = AutoTokenizer.from_pretrained(modelo_carpeta)
except Exception as e:
    print("No se encontró el tokenizer en el modelo entrenado, usando el tokenizer de distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

modelo = DistilBertForSequenceClassification.from_pretrained(modelo_carpeta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

def limpiar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    texto = texto.lower().strip()
    return texto

def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    inputs = tokenizer(
        texto_limpio,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    logits = outputs.logits
    prediccion = torch.argmax(logits, dim=1).item()
    
    mapeo_sentimientos = {0: "Negativo", 1: "Neutral", 2: "Positivo"}
    return mapeo_sentimientos[prediccion]

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        texto = request.form["texto"]
        resultado = predecir_sentimiento(texto)
    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)

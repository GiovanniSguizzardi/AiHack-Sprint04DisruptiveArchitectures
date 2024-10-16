from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import datetime
import cohere

# Chave API do cliente Cohere
cohere_client = cohere.Client('P6VLNucYpnxTdtIuaPJnIyzeTaicFZ9zmHYi9nVe')

app = Flask(__name__)

# Função para gerar relatório detalhado com a IA da Cohere
def gerar_relatorio_ia_generativa(classe_prevista):
    response = cohere_client.generate(
        model='command-xlarge-nightly',  # Use o modelo mais apropriado
        prompt=f"Gerar um relatório médico detalhado sobre a condição {classe_prevista}.",
        max_tokens=300
    )
    return response.generations[0].text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keras', methods=['POST'])
def uploadKeras():
    file = request.files['imagem']

    # Caminho para o modelo e rótulos
    model_path = r"C:\Users\giova\Downloads\Sprint 4 - IA\DataSet-ia\modelos\keras_model.h5"
    labels_path = r"C:\Users\giova\Downloads\Sprint 4 - IA\DataSet-ia\modelos\labels.txt"

    # Carregar o modelo
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        return {'error': f'Error loading model: {str(e)}'}, 500

    # Carregar os rótulos
    try:
        with open(labels_path, "r") as f:
            class_names = f.readlines()
        class_names = [name.strip() for name in class_names]
    except Exception as e:
        return {'error': f'Error loading labels: {str(e)}'}, 500

    # Criar o array da forma certa para alimentar o modelo Keras
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Processar a imagem
    image = Image.open(file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converter para numpy array
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    # Obter o índice da classe com a maior probabilidade
    if prediction.shape[1] == len(class_names):
        index = np.argmax(prediction[0])
        class_name = class_names[index] if index < len(class_names) else "Unknown"
        confidence_score = prediction[0][index]
    else:
        class_name = "Unknown"
        confidence_score = 0.0

    # Gerar relatórios
    relatorio_ia = gerar_relatorio_ia_generativa(class_name)
    data_atual = datetime.datetime.now()
    horario_atual = data_atual.strftime('%d/%m/%Y')
    hora_atual = data_atual.strftime('%H:%M:%S')

    # Renderizar o template HTML para mostrar na web
    return render_template('resultado.html',
                           class_name=class_name,
                           confidence=f'{confidence_score:.2f}',
                           data=horario_atual,
                           time=hora_atual,
                           relatorio_ia=relatorio_ia)

if __name__ == '__main__':
    app.run(debug=True)
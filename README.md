# Aplicação de IA para Diagnóstico de Imagens Médicas

Este projeto utiliza um modelo de Inteligência Artificial (IA) para análise de imagens médicas de pulmão e gera relatórios detalhados utilizando uma IA generativa.

## Funcionalidades

- **Classificação de imagens médicas**: O modelo de IA detecta condições como pneumonia, Covid-19 ou pulmões saudáveis a partir de imagens de raios-X.
- **Geração de relatórios**: Com base na previsão da IA, um relatório médico detalhado é gerado utilizando uma IA generativa (Cohere).
- **Interface amigável**: Permite o upload de imagens e exibe o resultado da previsão juntamente com um relatório detalhado.

## Tecnologias Utilizadas

- **Flask**: Framework web para backend.
- **TensorFlow e Keras**: Para o modelo de reconhecimento de imagens.
- **Cohere API**: Para geração de relatórios médicos via IA generativa.
- **PIL (Python Imaging Library)**: Para processamento de imagens.

## Como Configurar o Projeto Localmente

### Pré-requisitos

- **Python 3.8+** instalado
- Bibliotecas Python:
    ```bash
    pip install flask tensorflow pillow cohere
    ```

### Passos

1. Clone o repositório:
    ```bash
    git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
    cd SEU_REPOSITORIO
    ```

2. Coloque seu modelo treinado (arquivo `.h5`) e os rótulos (arquivo `.txt`) na pasta indicada no projeto (por exemplo, `DataSet-ia/`).

3. Coloque a chave da API da Cohere no código Python em `api.py`:
    ```python
    cohere_client = cohere.Client('SUA_API_KEY_AQUI')
    ```

4. Execute o servidor Flask:
    ```bash
    python api.py
    ```

5. Acesse a aplicação no navegador:
    ```
    http://127.0.0.1:5000/
    ```

## Estrutura do Projeto

```plaintext
Sprint 3 - IA/
├── DataSet-ia/             # Contém o modelo treinado (.h5) e rótulos (.txt)
├── templates/              # Arquivos HTML (interface do usuário)
│   ├── index.html          # Página de upload de imagem
│   ├── resultado.html      # Página de resultado e relatório
├── static/                 # Arquivos estáticos (imagens, CSS, etc.)
│   ├── img/                # Contém a logo da aplicação
│       └── aihack_logo.png
├── api.py                  # Arquivo principal com lógica da aplicação e IA
├── README.md               # Documentação do projeto

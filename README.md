# Redes Neurais Recorrentes — Formação Deep Learning com PyTorch

Este repositório reúne os notebooks trabalhados no curso **Redes Neurais Recorrentes** da formação *Deep Learning com PyTorch* (Alura). O material cobre desde a classificação de nomes com RNNs artesanais até um pipeline completo de análise de sentimentos com TorchText.

## Configuração do ambiente

1. (Opcional, mas recomendado) Crie e ative um ambiente virtual  
   `python -m venv .venv && source .venv/bin/activate` (Linux/macOS)  
   `.venv\Scripts\activate` (Windows).
2. Instale as dependências com `pip install -r requirements.txt`.
3. Baixe um modelo de idioma do spaCy antes de executar o notebook de sentimentos, por exemplo `python -m spacy download en_core_web_sm`, e ajuste `spacy.load('en_core_web_sm')` se necessário.
4. Abra os notebooks com `jupyter notebook` ou `jupyter lab`.

> Os dados de nomes por nacionalidade já estão em `data/`. O dataset IMDb e os vetores GloVe (6B, 100d) são baixados automaticamente pelo TorchText na primeira execução das células correspondentes; mantenha a conexão com a internet ativa.

## Conteúdo dos notebooks

- `01-ClassificacaoDeSequencias.ipynb`: constrói uma character-level RNN com `nn.RNNCell`, codificação one-hot e treinamento supervisionado para prever a nacionalidade associada a um nome próprio.
- `02-ClassificacaoDeSequencias-GRU.ipynb`: refatora o modelo anterior usando `nn.GRU`, aproveitando processamento em batch (`batch_first=True`) e o estado oculto final para classificação.
- `03-AnaliseDeSentimentos.ipynb`: implementa um fluxo de NLP com TorchText para classificar resenhas do IMDb, contemplando criação de vocabulário, embeddings GloVe, `BucketIterator`, empacotamento de sequências e avaliação do modelo.

## Sugestão de estudo

Percorra os notebooks na ordem numérica. A base construída com RNNs simples em 01 e 02 facilita a compreensão dos recursos mais avançados empregados na análise de sentimentos do notebook 03.

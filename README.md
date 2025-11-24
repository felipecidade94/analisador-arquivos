# Analisador de Arquivos

O Analisador de Arquivos é uma aplicação em Python que permite enviar arquivos para um banco PostgreSQL, extrair o conteúdo, indexar com embeddings e fazer perguntas em linguagem natural sobre cada arquivo.  
Além disso, ele gera gráficos prontos sobre o uso do sistema, executa consultas SQL customizadas e oferece uma interface gráfica amigável em Tkinter.

<div align="center">

![Status](https://img.shields.io/badge/status-estável-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.x-blue)
![Banco](https://img.shields.io/badge/DB-PostgreSQL-blueviolet)
![IA](https://img.shields.io/badge/IA-RAG%20%2B%20Groq-orange)

</div>

---

## Visão geral

A ideia do projeto é ter um "laboratório" de arquivos inteligente. Você faz upload de PDFs, textos, planilhas, CSV, Word ou Markdown, o sistema extrai o conteúdo, gera um resumo e cria um índice vetorial com FAISS.

Depois disso, você pode:

- selecionar um arquivo
- fazer perguntas em linguagem natural sobre o conteúdo
- receber respostas usando RAG, com LangChain, HuggingFace Embeddings e modelos da Groq

Tudo isso fica registrado em um banco PostgreSQL, junto com logs e métricas, e ainda dá para gerar gráficos com um clique.

A aplicação pode ser usada tanto em modo terminal quanto pela interface gráfica em Tkinter. 

---

## Principais funcionalidades

- Upload de arquivos com armazenamento binário no banco
- Extração de texto de múltiplos formatos:
  - pdf, docx, xlsx, xls, csv, txt, md :contentReference[oaicite:2]{index=2}
- Geração de resumo automático de cada arquivo com LLM (ou resumo simples se a API não estiver configurada)
- Indexação do conteúdo com FAISS e sentence-transformers
- Perguntas em linguagem natural sobre um arquivo específico usando RAG
- Registro de perguntas, respostas e tempo de execução
- Logs detalhados de erros, uploads, resumos e indexação
- Remoção completa de arquivos e dados relacionados
- Execução de consultas SQL customizadas com exportação do resultado para Excel
- Gráficos prontos com Matplotlib:
  - quantidade de arquivos por tipo
  - perguntas por arquivo
  - tempo médio de resposta por tipo de arquivo :contentReference[oaicite:3]{index=3}
- Interface gráfica em Tkinter com:
  - chat com balões para usuário e IA
  - seleção de arquivo para perguntas
  - botões para criar e remover tabelas, upload, gráficos, SQL customizado e remoção de arquivos :contentReference[oaicite:4]{index=4}

---

## Tecnologias utilizadas

- Python 3.x
- PostgreSQL
- SQLAlchemy (ORM)
- LangChain
- FAISS
- sentence-transformers (all-MiniLM-L6-v2)
- Groq (ChatGroq)
- PyMuPDF (fitz) para PDF
- python-docx para Word
- pandas para manipulação de dados
- matplotlib para gráficos
- markdown2 para tratamento de Markdown
- Tkinter e tkinterweb para a interface gráfica e renderização de Markdown
- python-dotenv para carregar variáveis de ambiente 

---

## Arquitetura do projeto

O projeto é dividido em dois módulos principais:

- `main.py`: camada de domínio, banco de dados e IA  
  - modelos SQLAlchemy
  - extração de conteúdo dos arquivos
  - embeddings e FAISS
  - RAG para responder perguntas com base no arquivo
  - criação, drop e verificação de tabelas
  - upload, remoção e log
  - geração de gráficos e consultas SQL customizadas
  - menu opcional de terminal :contentReference[oaicite:6]{index=6}

- `gui.py`: interface gráfica  
  - janela principal em Tkinter
  - visual de chat com balões para usuário, IA e sistema
  - botões para:
    - criar tabelas
    - remover tabelas
    - remover arquivo
    - listar arquivos
    - upload
    - perguntar sobre arquivo
    - rodar 3 gráficos prontos
    - executar consulta SQL customizada
    - sair
  - uso de HtmlFrame para exibir resposta da IA em Markdown no chat :contentReference[oaicite:7]{index=7}

Pastas e arquivos gerados automaticamente:

- `indices_faiss`: índices vetoriais FAISS por arquivo
- `charts`: imagens dos gráficos gerados
- `consultas`: resultados de consultas SQL customizadas em Excel :contentReference[oaicite:8]{index=8}

---

## Configuração

### Variáveis de ambiente

O projeto utiliza um arquivo `.env` para configurar:

```text
DATABASE_URL=postgresql+psycopg2://usuario:senha@host:5432/nome_do_banco
GROQ_API_KEY=sua_chave_da_groq_aqui
GROQ_API_MODEL=nome_do_modelo_groq
````

Exemplos de modelo que você pode usar:

* `llama-3.3-70b-versatile`
* `llama-3.1-8b-instant`
* outros modelos suportados pela Groq

Se `GROQ_API_KEY` não estiver definido, o sistema ainda funciona para upload, extração e logs, mas a parte de IA retorna respostas simplificadas com base no contexto recuperado. 

---

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/cidade-felipe/analisador-arquivos.git
cd analisador-arquivos
```

### 2. Criar ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.\.venv\Scripts\activate    # Windows
```

### 3. Instalar dependências

Se você tiver o `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

Caso contrário, instale pelo menos:

```bash
pip install python-dotenv sqlalchemy psycopg2-binary pandas matplotlib faiss-cpu \
langchain langchain-community langchain-huggingface langchain-groq pymupdf python-docx \
markdown2 tkinterweb
```

A instalação de Tkinter varia conforme o sistema operacional, em muitos casos ele já vem com o Python.

### 4. Executar a interface gráfica

```bash
python gui.py
```

A interface vai abrir com:

* um painel de chat à esquerda
* o menu de botões à direita
* e uma mensagem inicial do sistema informando que a interface foi carregada com sucesso 

### 5. Executar pelo terminal (opcional)

Se preferir usar o menu em modo texto:

```bash
python main.py
```

Ele oferece opções para criar tabelas, remover, fazer upload, perguntar sobre arquivos, gerar gráficos e rodar consultas SQL. 

---

## Fluxo típico de uso na interface gráfica

1. Abrir o programa com `python gui.py`
2. Clicar em "Criar tabelas" na primeira execução
3. Usar "Upload de arquivo" para enviar um PDF, CSV, DOCX, TXT, Excel ou Markdown
4. Clicar em "Perguntar sobre um arquivo" e escolher o arquivo na lista
5. Digitar uma pergunta na caixa de texto e apertar Enter ou clicar em "Enviar"
6. Ver a resposta da IA no chat, com formatação em Markdown
7. Usar os outros botões conforme necessário:

   * "3 Gráficos prontos" para visualizar estatísticas
   * "Consulta SQL customizada" para gerar consultas em Excel
   * "Remover arquivo" ou "Remover tabelas" conforme necessidade

---

## Gráficos disponíveis

Os três gráficos prontos gerados por `3 Gráficos prontos` são:

1. Quantidade de arquivos por tipo
2. Número de perguntas por arquivo
3. Tempo médio de resposta por tipo de arquivo

Eles são exibidos na tela e salvos na pasta `charts`. 

---

## Possíveis melhorias

Algumas ideias para evolução futura:

* Dashboard web para visualizar gráficos e consultas
* Filtros por usuário, data e tipo de arquivo
* Histórico de conversas por arquivo
* Configuração da aplicação via arquivo YAML ou JSON
* Testes automatizados para as funções principais

---

## Autor

Felipe Cidade

---

## Licença

Este projeto está licenciado sob a licença MIT.
Consulte o arquivo `LICENSE` para mais detalhes.
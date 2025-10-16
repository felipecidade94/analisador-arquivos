# Analisador de Arquivos com IA e Banco de Dados

Este projeto implementa um sistema inteligente de análise de arquivos utilizando **Python, SQLAlchemy, PostgreSQL e Groq (LangChain)**.
Ele foi desenvolvido como trabalho final da disciplina **Banco de Dados (DEC7588 – UFSC 2025.2)** e integra conceitos de banco relacional, processamento de dados, inteligência artificial e visualização de resultados.

---

## 1. Objetivo do Sistema

O sistema tem como propósito **analisar, armazenar e gerar insights automáticos** sobre arquivos de diferentes formatos, como PDF, DOCX, XLSX, CSV e TXT.
Cada arquivo é processado, indexado semanticamente e integrado a uma IA generativa (via Groq API), permitindo consultas e perguntas inteligentes sobre o conteúdo.

Além disso, o sistema gera **resumos automáticos** dos arquivos e consultas com **gráficos dinâmicos** baseados em dados armazenados no banco.

---

## 2. Funcionalidades Principais

* **Upload de arquivos** com armazenamento binário (BYTEA) no PostgreSQL
* **Extração automática de texto** de PDFs, planilhas e documentos
* **Indexação semântica** via **FAISS + Sentence Transformers**
* **Integração com IA Groq** para responder perguntas sobre os arquivos
* **Geração automática de resumos** de cada arquivo
* **Consultas SQL customizadas** e **gráficos automáticos**
* **Logs detalhados** de todas as operações realizadas

---

## 3. Estrutura do Banco de Dados

O sistema utiliza 10 entidades principais:

1. **tipo_arquivo** – define a extensão e categoria dos arquivos
2. **arquivo** – armazena metadados e conteúdo binário
3. **conteudo_extraido** – texto processado de cada arquivo
4. **embedding** – metadados dos vetores semânticos gerados
5. **pergunta** – histórico de perguntas feitas à IA
6. **resposta_ia** – respostas fornecidas pelo modelo Groq
7. **consulta_sql** – consultas executadas pelo usuário
8. **grafico** – gráficos gerados a partir das consultas
9. **log_sistema** – registro de eventos e erros
10. **resumo** – resumo automático do conteúdo de cada arquivo

---

## 4. Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Banco de Dados:** PostgreSQL
* **ORM:** SQLAlchemy
* **IA Generativa:** LangChain + Groq API
* **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
* **Busca Vetorial:** FAISS
* **Visualização:** Matplotlib
* **Extração de Conteúdo:** PyMuPDF, python-docx, pandas, openpyxl
* **Ambiente:** python-dotenv, psycopg2

---

## 5. Instalação

### Passo 1: Clonar o repositório

```bash
git clone https://github.com/seuusuario/analisador-arquivos.git
cd analisador-arquivos
```

### Passo 2: Criar e ativar o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\Scripts\\activate    # Windows
```

### Passo 3: Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 4: Criar o arquivo `.env`

```bash
DATABASE_URL=postgresql+psycopg2://usuario:senha@localhost:5432/analisador
GROQ_API_KEY=sua_chave_aqui
GROQ_API_MODEL=llama-3.3-70b-versatile
```

---

## 6. Execução do Sistema

```bash
python analisador_de_arquivos.py
```

### Menu Principal

```
============== MENU ==============
1) Criar tabelas
2) Remover tabelas
3) Upload de arquivo
4) Perguntar sobre um arquivo
5) Consultas e gráficos prontos (3 exemplos)
6) Rodar consulta SQL customizada
0) Sair
```

### Exemplos de uso

* **Upload:** envia um arquivo PDF ou DOCX e gera automaticamente seu resumo.
* **Perguntar:** faz perguntas sobre o conteúdo do arquivo.
* **Consultas:** executa consultas SQL no banco e gera gráficos automáticos.

---

## 7. Estrutura de Pastas

```
analisador-arquivos/
├── analisador_de_arquivos.py
├── requirements.txt
├── .env
├── indices_faiss/
├── charts/
├── consultas/
└── README.md
```

---

## 8. Consultas e Gráficos

O sistema gera três consultas automáticas (exibindo e salvando gráficos):

* Quantidade de arquivos por tipo
* Número de perguntas por arquivo
* Tempo médio de resposta da IA por tipo de arquivo

Além disso, o usuário pode realizar **consultas personalizadas** e salvar os resultados em **planilhas Excel**.

---

## 9. Licença

Este projeto é distribuído sob a licença MIT.
Sinta-se livre para estudar, modificar e adaptar o código para seus próprios projetos acadêmicos ou pessoais.

---

## 10. Créditos

Desenvolvido por **Felipe Cidade Soares**
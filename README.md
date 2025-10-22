

````markdown
# 🧠 Analisador de Arquivos com IA e SQLAlchemy

Sistema Python para gerenciamento, extração e análise de arquivos utilizando **SQLAlchemy**, **LangChain**, e **FAISS**.  
O projeto permite **upload de documentos**, **armazenamento em banco de dados**, **extração de texto**, **resumos automáticos**, **criação de embeddings vetoriais** e **consultas SQL com visualização gráfica**.

---

## ⚙️ Funcionalidades Principais

- **Upload inteligente**  
  - Detecção de arquivos duplicados via hash SHA256  
  - Extração automática de texto (PDF, DOCX, CSV, Excel, TXT e Markdown)  
  - Geração de resumo inicial e logs detalhados  

- **Banco de dados relacional (SQLAlchemy ORM)**  
  - Estrutura robusta com entidades relacionadas (`Arquivo`, `ConteudoExtraido`, `Embedding`, `Resumo`, `Pergunta`, `RespostaIA`, `Log`, `ConsultaSQL`, `ResultadoConsulta`)  
  - Suporte a consultas híbridas ORM + SQL bruto  

- **Consultas SQL customizadas e exportação**  
  - Execução livre de comandos `SELECT`  
  - Exportação de resultados em `.xlsx`  
  - Diretório automático `/consultas` para histórico  

- **Visualização de dados com gráficos (Matplotlib)**  
  - 3 gráficos prontos:
    1. Arquivos por tipo  
    2. Perguntas por arquivo (nomes truncados para 15 caracteres)  
    3. Tempo médio de resposta por tipo de arquivo  
  - Eixos formatados e rotação automática das legendas  

- **Integração com embeddings (FAISS + LangChain)**  
  - Fragmentação de texto (`RecursiveCharacterTextSplitter`)  
  - Armazenamento e reuso de índices FAISS locais  

---

## 🧩 Diagrama de Classes (Mermaid)

```mermaid
classDiagram
    class TipoArquivo {
        +id : int
        +nome : str
    }
    class Arquivo {
        +id : int
        +nome : str
        +tamanho : int
        +hash_sha256 : str
        +data_upload : datetime
    }
    class ConteudoExtraido {
        +id : int
        +texto : text
    }
    class Embedding {
        +id : int
        +num_chunks : int
        +dim : int
        +index_path : str
    }
    class Resumo {
        +id : int
        +texto : text
    }
    class Pergunta {
        +id : int
        +texto : text
        +data : datetime
    }
    class RespostaIA {
        +id : int
        +resposta : text
        +tempo_execucao : float
    }
    class Log {
        +id : int
        +acao : str
        +detalhe : text
        +data : datetime
    }
    class ConsultaSQL {
        +id : int
        +sql : text
        +descricao : text
    }
    class ResultadoConsulta {
        +id : int
        +caminho_arquivo : str
        +dados_json : JSON
    }

    TipoArquivo "1" --> "N" Arquivo
    Arquivo "1" --> "1" ConteudoExtraido
    ConteudoExtraido "1" --> "N" Embedding
    Arquivo "1" --> "1" Resumo
    Arquivo "1" --> "N" Pergunta
    Pergunta "1" --> "1" RespostaIA
    Arquivo "1" --> "N" Log
    ConsultaSQL "1" --> "1" ResultadoConsulta
````

---

## 🧰 Estrutura de Pastas

```
analisador-arquivos/
│
├── main.py
├── interface.py
├── .env
├── .gitignore
├── requirements.txt
│
├── indices_faiss/        # Índices vetoriais FAISS
├── charts/               # Gráficos gerados
├── consultas/            # Resultados de SQL customizados
└── venv/                 # Ambiente virtual (ignorado no Git)
```

---

## 🚀 Execução

### 1. Crie e ative o ambiente virtual

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o `.env`

Crie um arquivo `.env` na raiz do projeto:

```
DATABASE_URL=sqlite:///analisador.db
GROQ_API_KEY=sua_chave_aqui
GROQ_API_MODEL=llama-3-70b
```

### 4. Execute o sistema

```bash
python main.py
```

O menu interativo CLI será exibido com opções para criar tabelas, enviar arquivos, consultar, ou gerar gráficos.

---

## 🧠 Boas Práticas

* Sempre use `venv` para isolar dependências.
* Mantenha o arquivo `.env` fora do controle de versão.
* Limpe os diretórios `indices_faiss/`, `charts/` e `consultas/` antes de novos testes.
* Commits devem conter mensagens claras (`feat:`, `fix:`, `refactor:`, etc).
* Faça backup regular do banco de dados local.

---

## 📜 Licença

Projeto sob licença MIT.
Desenvolvido por **Felipe Cidade Soares**.

````
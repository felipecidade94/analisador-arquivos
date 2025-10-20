# Analisador de Arquivos com IA e Interface Gráfica

Este projeto é um **analisador inteligente de arquivos** capaz de ler, armazenar, resumir e responder perguntas sobre documentos em diferentes formatos — como PDF, DOCX, XLSX, CSV, TXT e Markdown — utilizando **LangChain**, **Groq** e **SQLAlchemy**.  
O sistema inclui uma **interface gráfica (Tkinter)** que facilita a interação com o usuário e a visualização dos resultados.

---

## Funcionalidades Principais

- **Upload e processamento automático de arquivos**  
  Ao enviar um arquivo, o sistema:
  1. Extrai o conteúdo textual (com suporte a múltiplos formatos);
  2. Gera embeddings e um índice FAISS para consultas semânticas;
  3. Produz automaticamente um resumo via modelo Groq;
  4. Realiza uma análise técnica do arquivo (palavras, páginas, sentimento, densidade, etc.);
  5. Registra todas as ações e erros na tabela de log.

- **Consultas e respostas automáticas via IA**  
  É possível fazer perguntas sobre qualquer arquivo enviado.  
  O sistema busca os trechos mais relevantes no conteúdo e responde de forma contextualizada.

- **Consultas SQL e geração de gráficos**  
  O usuário pode:
  - Rodar consultas SQL customizadas;
  - Executar gráficos prontos (quantidade de arquivos, perguntas por documento, tempo médio de resposta);
  - Visualizar resultados salvos automaticamente em planilhas Excel.

- **Interface Gráfica (Tkinter)**  
  A interface inclui:
  - Área de chat com histórico de mensagens;
  - Botões laterais para operações principais (upload, gráficos, consultas);
  - Campo de entrada e envio de perguntas com exibição de respostas;
  - Logotipo customizável e mensagens coloridas para feedback visual.

---

## Estrutura do Banco de Dados

O banco foi modelado com **SQLAlchemy ORM**, contendo 11 entidades principais:

```mermaid
erDiagram
    TipoArquivo ||--o{ Arquivo : possui
    Arquivo ||--|| ConteudoExtraido : tem
    ConteudoExtraido ||--o{ Embedding : gera
    Arquivo ||--o{ Pergunta : recebe
    Pergunta ||--|| RespostaIA : gera
    Arquivo ||--o{ Log : registra
    Arquivo ||--|| Resumo : sintetiza
    Arquivo ||--|| AnaliseArquivo : analisa
    ConsultaSQL ||--|| ResultadoConsulta : produz
````

**Descrição resumida das entidades:**

* **Arquivo**: armazena o binário e metadados de cada documento.
* **ConteudoExtraido**: guarda o texto puro do arquivo.
* **Embedding**: metadados sobre o índice vetorial FAISS.
* **Pergunta** e **RespostaIA**: controlam o diálogo entre o usuário e o sistema.
* **Resumo**: resumo textual gerado automaticamente via IA.
* **Log**: histórico detalhado de operações e erros.
* **AnaliseArquivo**: estatísticas e insights automáticos (palavras, tamanho, sentimento).
* **ConsultaSQL** e **ResultadoConsulta**: consultas salvas e seus respectivos resultados.

---

## Tecnologias Utilizadas

* **Python 3.12+**
* **LangChain**, **LangChain-Text-Splitters**, **LangChain-Groq**
* **SQLAlchemy**
* **Pandas**
* **Matplotlib**
* **Tkinter (GUI)**
* **PyMuPDF (PDF)**, **python-docx**, **markdown2**
* **HuggingFaceEmbeddings**
* **FAISS** (armazenamento vetorial)
* **dotenv** (configuração do ambiente)

---

## Boas Práticas e Execução

### 1. Criar e ativar o ambiente virtual (venv)

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente (`.env`)

Crie um arquivo `.env` na raiz do projeto:

```
DATABASE_URL=sqlite:///analisador.db
GROQ_API_KEY=sua_chave_aqui
GROQ_API_MODEL=llama-3.3-70b-versatile
```

### 4. Executar o programa

```bash
python main.py
```

ou, para rodar a interface gráfica:

```bash
python interface.py
```

---

## Boas Práticas de Desenvolvimento

* Utilize **venv** sempre que iniciar um novo ambiente — isso evita conflitos de dependências.
* Prefira **métodos ORM (SQLAlchemy)** a SQL direto para manter portabilidade e segurança.
* Evite hardcodes de caminhos: use `os.path.join()` e variáveis do `.env`.
* Faça **commits frequentes** e documente mudanças relevantes.
* Teste cada tipo de arquivo suportado antes de subir alterações.

---

## Estrutura do Projeto

```
analisador-arquivos/
├── main.py                # Núcleo do sistema e lógica principal
├── interface.py           # Interface gráfica Tkinter
├── requirements.txt       # Dependências
├── .env                   # Configurações locais (não versionar)
├── charts/                # Gráficos gerados
├── consultas/             # Consultas SQL salvas
├── indices_faiss/         # Índices vetoriais FAISS
└── tests/                 # Recursos auxiliares (logos, arquivos de teste, etc.)
```

---

## Contribuição

Contribuições são bem-vindas.
Antes de enviar um pull request:

1. Certifique-se de que todas as dependências estão atualizadas;
2. Teste as principais funções (`upload`, `pergunta`, `consultas`);
3. Documente novas entidades ou alterações estruturais no README.

---

## Licença

Este projeto é distribuído sob a licença MIT.
Você pode usar, modificar e distribuir livremente, desde que mantenha os créditos originais.

## Créditos

Desenvolvido por **Felipe Cidade Soares**

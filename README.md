## Sistema Inteligente de Análise de Arquivos
-----------------------------------------

- Faz upload de arquivos (PDF, DOCX, XLSX, CSV, TXT), guarda o binário em PostgreSQL (BYTEA)
- Extrai conteúdo textual (quando aplicável) e indexa em embeddings (FAISS + sentence-transformers)
- Usa Groq (ChatGroq via LangChain) para responder perguntas com contexto recuperado por similaridade
- Permite rodar consultas SQL de agregação sobre tabelas do próprio sistema e gerar gráficos (PNG)
- Mantém logs, perguntas e respostas, gráficos e metadados no banco (10 entidades)

Requisitos: Python 3.10+

Instale dependências (exemplo):
    pip install python-dotenv sqlalchemy psycopg2-binary langchain groq langchain-groq
    pip install sentence-transformers faiss-cpu numpy pandas matplotlib python-docx pymupdf openpyxl

Config .env (exemplo):
    DATABASE_URL=postgresql+psycopg2://usuario:senha@localhost:5432/analisador
    GROQ_API_KEY=...  # sua chave
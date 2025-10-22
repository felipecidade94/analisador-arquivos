# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import io
import json
import time
import hashlib
import datetime as dt
from typing import Optional, Tuple

from dotenv import load_dotenv
import shutil

# Banco de dados
from sqlalchemy import create_engine, text, Column, Integer, String, Text, LargeBinary, DateTime, ForeignKey, Float, JSON, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Extração de conteúdo
import pandas as pd
import markdown2
import re

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import docx  # python-docx
except Exception:
    docx = None

# IA / Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Visualização
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_MODEL = os.getenv('GROQ_API_MODEL')

if not DATABASE_URL:
    raise RuntimeError('Defina DATABASE_URL no .env')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ------------------------------------------------------------
# Modelos ORM (mantidos)
# ------------------------------------------------------------
class TipoArquivo(Base):
    __tablename__ = 'tipo_arquivo'
    id = Column(Integer, primary_key=True)
    nome = Column(String(50), unique=True, nullable=False)
    arquivos = relationship('Arquivo', back_populates='tipo')

class Arquivo(Base):
    __tablename__ = 'arquivo'
    id = Column(Integer, primary_key=True)
    nome = Column(String(255))
    tipo_id = Column(Integer, ForeignKey('tipo_arquivo.id'))
    data_upload = Column(DateTime, default=func.now())
    conteudo = Column(LargeBinary)
    tamanho = Column(Integer)
    hash_sha256 = Column(String(64), index=True)
    tipo = relationship('TipoArquivo', back_populates='arquivos')
    conteudo_extraido = relationship('ConteudoExtraido', back_populates='arquivo', uselist=False)
    resumo = relationship('Resumo', back_populates='arquivo', uselist=False)
    perguntas = relationship('Pergunta', back_populates='arquivo')
    logs = relationship('Log', back_populates='arquivo')

class ConteudoExtraido(Base):
    __tablename__ = 'conteudo_extraido'
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey('arquivo.id'), unique=True)
    texto = Column(Text)
    arquivo = relationship('Arquivo', back_populates='conteudo_extraido')
    embeddings = relationship('Embedding', back_populates='conteudo')

class Embedding(Base):
    __tablename__ = 'embedding'
    id = Column(Integer, primary_key=True)
    conteudo_id = Column(Integer, ForeignKey('conteudo_extraido.id'))
    num_chunks = Column(Integer)
    dim = Column(Integer)
    index_path = Column(String(255))
    conteudo = relationship('ConteudoExtraido', back_populates='embeddings')

class Pergunta(Base):
    __tablename__ = 'pergunta'
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey('arquivo.id'))
    texto = Column(Text)
    data = Column(DateTime, default=func.now())
    arquivo = relationship('Arquivo', back_populates='perguntas')
    resposta = relationship('RespostaIA', back_populates='pergunta', uselist=False)

class RespostaIA(Base):
    __tablename__ = 'resposta_ia'
    id = Column(Integer, primary_key=True)
    pergunta_id = Column(Integer, ForeignKey('pergunta.id'))
    resposta = Column(Text)
    tempo_execucao = Column(Float)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    pergunta = relationship('Pergunta', back_populates='resposta')

class Log(Base):
    __tablename__ = 'log'
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey('arquivo.id'))
    acao = Column(String(100))
    detalhe = Column(Text)
    data = Column(DateTime, default=func.now())
    arquivo = relationship('Arquivo', back_populates='logs')

class Resumo(Base):
    __tablename__ = 'resumo'
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey('arquivo.id'), unique=True)
    texto = Column(Text)
    data = Column(DateTime, default=func.now())
    arquivo = relationship('Arquivo', back_populates='resumo')

class ConsultaSQL(Base):
    __tablename__ = 'consulta_sql'
    id = Column(Integer, primary_key=True)
    sql = Column(Text)
    descricao = Column(Text)
    data = Column(DateTime, default=func.now())
    resultado = relationship('ResultadoConsulta', back_populates='consulta', uselist=False)

class ResultadoConsulta(Base):
    __tablename__ = 'resultado_consulta'
    id = Column(Integer, primary_key=True)
    consulta_id = Column(Integer, ForeignKey('consulta_sql.id'), unique=True)
    caminho_arquivo = Column(String(255))
    dados_json = Column(JSON)
    linhas = Column(Integer)
    colunas = Column(Integer)
    data_execucao = Column(DateTime, default=func.now())
    consulta = relationship('ConsultaSQL', back_populates='resultado')

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_DIR = 'indices_faiss'
CHART_DIR = 'charts'
RESULT_CONSULTA_DIR = 'consultas'
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RESULT_CONSULTA_DIR, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def infer_tipo_from_name(name: str) -> str:
    n = name.lower()
    for ext in ('.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.md'):
        if n.endswith(ext):
            return ext[1:]
    return 'desconhecido'

# ------------------------------------------------------------
# Extração de conteúdo (igual à original)
# ------------------------------------------------------------
def extract_text_from_pdf(data: bytes) -> str:
    if not fitz: raise RuntimeError('PyMuPDF não instalado')
    with fitz.open(stream=data, filetype='pdf') as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(data: bytes) -> str:
    bio = io.BytesIO(data)
    d = docx.Document(bio)
    return "\n".join(p.text for p in d.paragraphs)

def extract_text_from_csv(data: bytes) -> str:
    bio = io.BytesIO(data)
    df = pd.read_csv(bio)
    return df.to_csv(index=False)

def extract_text_from_excel(data: bytes) -> str:
    bio = io.BytesIO(data)
    dfs = pd.read_excel(bio, sheet_name=None)
    return "\n\n".join(f"# {k}\n{v.to_markdown(index=False)}" for k, v in dfs.items())

def extract_text_from_txt(data: bytes) -> str:
    return data.decode('utf-8', errors='ignore')

def extract_text_from_md(data: bytes) -> str:
    html = markdown2.markdown(data.decode('utf-8', errors='ignore'))
    return re.sub(r'<[^>]+>', '', html).strip()

def extract_content_by_type(tipo: str, data: bytes) -> str:
    if tipo == 'pdf': return extract_text_from_pdf(data)
    if tipo == 'docx': return extract_text_from_docx(data)
    if tipo in {'xlsx','xls'}: return extract_text_from_excel(data)
    if tipo == 'csv': return extract_text_from_csv(data)
    if tipo == 'txt': return extract_text_from_txt(data)
    if tipo == 'md': return extract_text_from_md(data)
    return ''

# ------------------------------------------------------------
# Funções híbridas (SQL + ORM)
# ------------------------------------------------------------
def ensure_tipo(conn, nome_tipo: str) -> int:
    r = conn.execute(text("SELECT id FROM tipo_arquivo WHERE nome = :n"), {"n": nome_tipo}).fetchone()
    if r:
        return r[0]
    conn.execute(text("INSERT INTO tipo_arquivo (nome) VALUES (:n)"), {"n": nome_tipo})
    # ❌ NÃO usar conn.commit() dentro de um engine.begin()
    return conn.execute(text("SELECT id FROM tipo_arquivo WHERE nome = :n"), {"n": nome_tipo}).scalar()


def build_or_load_index_for_file(sess, conteudo: ConteudoExtraido) -> Tuple[FAISS, str]:
    texto = conteudo.texto or ''
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents([texto])
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    path = os.path.join(INDEX_DIR, f"faiss_{conteudo.arquivo_id}.index")
    if os.path.exists(path):
        vs = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(path)
    return vs, path

def upload_file(filepath: str) -> int:
    with engine.begin() as conn:
        data = open(filepath, "rb").read()
        nome = os.path.basename(filepath)
        tipo_nome = infer_tipo_from_name(nome)
        tipo_id = ensure_tipo(conn, tipo_nome)
        h = sha256_bytes(data)

        ex = conn.execute(text("SELECT id FROM arquivo WHERE hash_sha256 = :h"), {"h": h}).fetchone()
        if ex:
            conn.execute(text("INSERT INTO log (arquivo_id, acao, detalhe) VALUES (:id,'upload_duplicado',:n)"),
                         {"id": ex[0], "n": nome})
            return ex[0]

        conn.execute(text("""
            INSERT INTO arquivo (nome, tipo_id, conteudo, tamanho, hash_sha256)
            VALUES (:n,:t,:c,:tam,:h)
        """), {"n": nome, "t": tipo_id, "c": data, "tam": len(data), "h": h})

        arq_id = conn.execute(text("SELECT id FROM arquivo WHERE hash_sha256=:h"), {"h": h}).scalar()

        # Extração
        try:
            texto = extract_content_by_type(tipo_nome, data)
        except Exception as e:
            texto = ''
            conn.execute(text("INSERT INTO log (arquivo_id, acao, detalhe) VALUES (:id,'erro_extracao',:d)"),
                         {"id": arq_id, "d": str(e)})

        conn.execute(text("INSERT INTO conteudo_extraido (arquivo_id, texto) VALUES (:id,:t)"),
                     {"id": arq_id, "t": texto})

        resumo = texto[:500] + "..." if len(texto) > 500 else texto
        conn.execute(text("INSERT INTO resumo (arquivo_id, texto) VALUES (:id,:t)"), {"id": arq_id, "t": resumo})
        conn.execute(text("INSERT INTO log (arquivo_id, acao, detalhe) VALUES (:id,'upload_ok',:n)"),
                     {"id": arq_id, "n": nome})
        return arq_id

def answer_question(arquivo_id: int, pergunta: str) -> str:
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO pergunta (arquivo_id, texto) VALUES (:a,:t)"), {"a": arquivo_id, "t": pergunta})
        pid = conn.execute(text("SELECT MAX(id) FROM pergunta WHERE arquivo_id=:a"), {"a": arquivo_id}).scalar()
        texto = conn.execute(text("SELECT texto FROM conteudo_extraido WHERE arquivo_id=:a"),
                             {"a": arquivo_id}).scalar() or ''

        resposta = f"[Contexto local]\n{texto[:800]}"
        conn.execute(text("""
            INSERT INTO resposta_ia (pergunta_id, resposta, tempo_execucao)
            VALUES (:p,:r,0.1)
        """), {"p": pid, "r": resposta})
        return resposta

def run_and_plot(titulo, eixo_x, eixo_y, sql, descricao, chart_type='barras'):
    df = pd.read_sql_query(sql, con=engine)
    if df.empty:
        print("[AVISO] Consulta vazia.")
        return

    # 🔹 Trunca nomes longos automaticamente (apenas para gráficos com nomes)
    if df.shape[1] >= 1 and df.iloc[:, 0].dtype == object:
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: str(x)[:15] + '…' if isinstance(x, str) and len(x) > 15 else x)

    plt.figure(figsize=(8, 5))
    x, y = df.iloc[:, 0], pd.to_numeric(df.iloc[:, 1], errors='coerce')

    if chart_type == 'linhas':
        plt.plot(x, y, marker='o', linewidth=2)
    elif chart_type == 'pizza':
        plt.pie(y, labels=x, autopct='%1.1f%%')
    else:
        plt.bar(x, y, color='#3478eb')

    plt.title(titulo, fontsize=12, fontweight='bold')
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def create_all():
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for t in ['pdf','docx','xlsx','xls','csv','txt','md']:
            ensure_tipo(conn, t)
    return "[OK] Tabelas criadas."


def drop_all():
    Base.metadata.drop_all(engine)
    for p in (INDEX_DIR, CHART_DIR, RESULT_CONSULTA_DIR):
        if os.path.exists(p): shutil.rmtree(p)
    return "[OK] Tabelas e pastas removidas."


# ------------------------------------------------------------
# Menu CLI atualizado
# ------------------------------------------------------------
MENU = """
============== MENU ==============
1) Criar tabelas
2) Remover tabelas
3) Upload de arquivo
4) Perguntar sobre um arquivo
5) 3 exemplos de gráficos prontos
6) Consulta SQL customizada
0) Sair
> """


def menu():
    while True:
        op = input(MENU).strip()

        # 1. Criar tabelas
        if op == '1':
            print(create_all())

        # 2. Remover tabelas
        elif op == '2':
            print(drop_all())

        # 3. Upload
        elif op == '3':
            path = input("Caminho do arquivo: ").strip('"')
            if os.path.exists(path):
                upload_file(path)
            else:
                print("[ERRO] Arquivo não encontrado.")

        # 4. Perguntar sobre arquivo
        elif op == '4':
            try:
                a_id = int(input("ID do arquivo: "))
            except ValueError:
                print("[ERRO] ID inválido.")
                continue
            pergunta = input("Pergunta: ")
            print(answer_question(a_id, pergunta))

        # 5. Consultas e gráficos prontos
        elif op == '5':
            print("\n[INFO] Gerando consultas e gráficos padrão...\n")

            sql1 = """
                SELECT t.nome AS tipo, COUNT(a.id) AS qtde
                FROM tipo_arquivo t
                LEFT JOIN arquivo a ON a.tipo_id = t.id
                GROUP BY t.nome ORDER BY qtde DESC
            """
            run_and_plot("Arquivos por tipo", "Tipo", "Quantidade", sql1, "Arquivos agrupados por tipo", "barras")

            sql2 = """
                SELECT a.nome AS arquivo, COUNT(p.id) AS perguntas
                FROM arquivo a
                LEFT JOIN pergunta p ON p.arquivo_id = a.id
                GROUP BY a.nome ORDER BY perguntas DESC
            """
            run_and_plot("Perguntas por arquivo", "Arquivo", "Qtd", sql2, "Número de perguntas por arquivo", "barras")

            sql3 = """
                SELECT t.nome AS tipo, COALESCE(AVG(r.tempo_execucao),0) AS tempo_medio_s
                FROM tipo_arquivo t
                LEFT JOIN arquivo a ON a.tipo_id = t.id
                LEFT JOIN pergunta p ON p.arquivo_id = a.id
                LEFT JOIN resposta_ia r ON r.pergunta_id = p.id
                GROUP BY t.nome ORDER BY tempo_medio_s DESC
            """
            run_and_plot("Tempo médio por tipo", "Tipo", "Tempo (s)", sql3, "Tempo médio de resposta", "linhas")

            print("\n[OK] Consultas e gráficos prontos concluídos.\n")

        # 6. Consulta customizada
        elif op == '6':
            sql = input("\nDigite uma consulta SQL (apenas SELECT): ").strip()
            if not sql.lower().startswith("select"):
                print("[ERRO] Apenas consultas SELECT são permitidas.")
                continue
            try:
                df = pd.read_sql_query(sql, con=engine)
                if df.empty:
                    print("[AVISO] Nenhum resultado encontrado.")
                    continue
                timestamp = int(time.time())
                path = os.path.join(RESULT_CONSULTA_DIR, f"consulta_{timestamp}.xlsx")
                df.to_excel(path, index=False)
                print(f"[OK] Consulta executada com sucesso. Resultado salvo em: {path}")
            except Exception as e:
                print(f"[ERRO] Falha ao executar consulta: {e}")

        # 0. Sair
        elif op == '0':
            print("Até mais!")
            break

        else:
            print("[ERRO] Opção inválida.")


if __name__ == '__main__':
    menu()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import io
import json
import time
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple

from dotenv import load_dotenv

# Banco de dados
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, LargeBinary, DateTime, ForeignKey,
    Float, JSON, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Extração de conteúdo
import pandas as pd

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# IA / Embeddings / RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Visualização
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_MODEL = os.getenv("GROQ_API_MODEL")

if not DATABASE_URL:
    raise RuntimeError("Defina DATABASE_URL no .env (ex: postgresql+psycopg2://user:pass@host:5432/db)")
if not GROQ_API_KEY:
    print("[AVISO] GROQ_API_KEY não definido. Funções de LLM ficarão limitadas até você configurar.")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ------------------------------------------------------------
# Modelos (10 entidades)
# ------------------------------------------------------------
class TipoArquivo(Base):
    __tablename__ = "tipo_arquivo"
    id = Column(Integer, primary_key=True)
    nome = Column(String(50), nullable=False, unique=True)
    arquivos = relationship("Arquivo", back_populates="tipo")


class Arquivo(Base):
    __tablename__ = "arquivo"
    id = Column(Integer, primary_key=True)
    nome = Column(String(255))
    tipo_id = Column(Integer, ForeignKey("tipo_arquivo.id"))
    data_upload = Column(DateTime, default=func.now())
    conteudo = Column(LargeBinary)
    tamanho = Column(Integer)
    hash_sha256 = Column(String(64), index=True)


    tipo = relationship("TipoArquivo", back_populates="arquivos")
    conteudo_extraido = relationship("ConteudoExtraido", back_populates="arquivo", uselist=False)
    perguntas = relationship("Pergunta", back_populates="arquivo")
    consultas = relationship("ConsultaSQL", back_populates="arquivo")
    logs = relationship("LogSistema", back_populates="arquivo")
    resumo = relationship("Resumo", back_populates="arquivo", uselist=False)


class ConteudoExtraido(Base):
    __tablename__ = "conteudo_extraido"
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey("arquivo.id"), unique=True)
    texto = Column(Text)


    arquivo = relationship("Arquivo", back_populates="conteudo_extraido")
    embeddings = relationship("Embedding", back_populates="conteudo")


class Embedding(Base):
    __tablename__ = "embedding"
    id = Column(Integer, primary_key=True)
    conteudo_id = Column(Integer, ForeignKey("conteudo_extraido.id"))
    num_chunks = Column(Integer)
    dim = Column(Integer)
    index_path = Column(String(255))


    conteudo = relationship("ConteudoExtraido", back_populates="embeddings")


class Pergunta(Base):
    __tablename__ = "pergunta"
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey("arquivo.id"))
    texto = Column(Text)
    data = Column(DateTime, default=func.now())


    arquivo = relationship("Arquivo", back_populates="perguntas")
    resposta = relationship("RespostaIA", back_populates="pergunta", uselist=False)


class RespostaIA(Base):
    __tablename__ = "resposta_ia"
    id = Column(Integer, primary_key=True)
    pergunta_id = Column(Integer, ForeignKey("pergunta.id"))
    resposta = Column(Text)
    tempo_execucao = Column(Float)
    tokens_input = Column(Integer, nullable=True)
    tokens_output = Column(Integer, nullable=True)


    pergunta = relationship("Pergunta", back_populates="resposta")


class ConsultaSQL(Base):
    __tablename__ = "consulta_sql"
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey("arquivo.id"), nullable=True)
    sql = Column(Text)
    descricao = Column(Text)
    data = Column(DateTime, default=func.now())


    arquivo = relationship("Arquivo", back_populates="consultas")
    graficos = relationship("Grafico", back_populates="consulta")


class Grafico(Base):
    __tablename__ = "grafico"
    id = Column(Integer, primary_key=True)
    consulta_id = Column(Integer, ForeignKey("consulta_sql.id"))
    tipo = Column(String(50))
    dados = Column(JSON)
    imagem_path = Column(String(255))


    consulta = relationship("ConsultaSQL", back_populates="graficos")


class LogSistema(Base):
    __tablename__ = "log_sistema"
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey("arquivo.id"), nullable=True)
    acao = Column(String(100))
    detalhe = Column(Text)
    data = Column(DateTime, default=func.now())


    arquivo = relationship("Arquivo", back_populates="logs")


class Resumo(Base):
    __tablename__ = "resumo"
    id = Column(Integer, primary_key=True)
    arquivo_id = Column(Integer, ForeignKey("arquivo.id"), unique=True)
    texto = Column(Text)
    data = Column(DateTime, default=func.now())


    arquivo = relationship("Arquivo", back_populates="resumo")



# ------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # rápido e bom
INDEX_DIR = "indices_faiss"
CHART_DIR = "charts"
RESULT_CONSULTA_DIR = 'consultas'
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(RESULT_CONSULTA_DIR, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def infer_tipo_from_name(name: str) -> str:
    n = name.lower()
    for ext in (".pdf", ".docx", ".xlsx", ".csv", ".txt"):
        if n.endswith(ext):
            return ext[1:]
    return "desconhecido"


def ensure_tipo(sess, nome_tipo: str) -> TipoArquivo:
    obj = sess.query(TipoArquivo).filter_by(nome=nome_tipo).one_or_none()
    if not obj:
        obj = TipoArquivo(nome=nome_tipo)
        sess.add(obj)
        sess.commit()
    return obj

# ------------------------------------------------------------
# Extração de conteúdo
# ------------------------------------------------------------

def extract_text_from_pdf(data: bytes) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF não instalado. pip install pymupdf")
    text = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text).strip()


def extract_text_from_docx(data: bytes) -> str:
    if not docx:
        raise RuntimeError("python-docx não instalado. pip install python-docx")
    bio = io.BytesIO(data)
    d = docx.Document(bio)
    return "\n".join(p.text for p in d.paragraphs).strip()


def extract_text_from_csv(data: bytes) -> str:
    bio = io.BytesIO(data)
    df = pd.read_csv(bio)
    # Representamos CSV como texto tabular simples para RAG
    return df.to_csv(index=False)


def extract_text_from_xlsx(data: bytes) -> str:
    bio = io.BytesIO(data)
    dfs = pd.read_excel(bio, sheet_name=None)  # todas as abas
    parts = []
    for sheet, df in dfs.items():
        parts.append(f"# Sheet: {sheet}\n" + df.to_csv(index=False))
    return "\n\n".join(parts)


def extract_text_from_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def extract_content_by_type(tipo: str, data: bytes) -> str:
    if tipo == "pdf":
        return extract_text_from_pdf(data)
    if tipo == "docx":
        return extract_text_from_docx(data)
    if tipo == "csv":
        return extract_text_from_csv(data)
    if tipo == "xlsx":
        return extract_text_from_xlsx(data)
    if tipo == "txt":
        return extract_text_from_txt(data)
    return ""

# ------------------------------------------------------------
# Embeddings / Índice por arquivo
# ------------------------------------------------------------

def build_or_load_index_for_file(sess, conteudo: ConteudoExtraido) -> Tuple[FAISS, str]:
    # Gera chunks e constrói um índice FAISS por arquivo
    texto = conteudo.texto or ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents([texto])

    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    index_path = os.path.join(INDEX_DIR, f"faiss_{conteudo.arquivo_id}.index")

    if os.path.exists(index_path):
        vs = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(index_path)

    # Persistimos metadados de Embedding
    meta = sess.query(Embedding).filter_by(conteudo_id=conteudo.id).one_or_none()
    if not meta:
        meta = Embedding(
            conteudo_id=conteudo.id,
            num_chunks=len(docs),
            dim=384,  # dimensão do all-MiniLM-L6-v2
            index_path=index_path,
        )
        sess.add(meta)
        sess.commit()
    return vs, index_path

# ------------------------------------------------------------
# Perguntas e respostas (Groq + RAG)
# ------------------------------------------------------------
SYSTEM_PROMPT = (
    "Você é um assistente que responde com base ESTRITA no contexto fornecido. "
    "Se a resposta não estiver no contexto, diga claramente que não encontrou. "
    "Seja conciso e cite trechos relevantes do contexto quando possível."
)

def answer_question(sess, arquivo_id: int, pergunta_texto: str) -> str:
    arq = sess.get(Arquivo, arquivo_id)
    if not arq or not arq.conteudo_extraido:
        raise ValueError("Arquivo ou conteúdo não encontrado para este ID.")

    # registrar pergunta
    q = Pergunta(arquivo_id=arquivo_id, texto=pergunta_texto)
    sess.add(q)
    sess.commit()

    start = time.time()

    # Recuperação semântica
    vs, _ = build_or_load_index_for_file(sess, arq.conteudo_extraido)
    retrieved_docs = vs.similarity_search(pergunta_texto, k=5)
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    if not GROQ_API_KEY:
        resposta_texto = "[GROQ_API_KEY ausente] Contexto recuperado:\n" + context[:1000]
        elapsed = time.time() - start
        r = RespostaIA(pergunta_id=q.id, resposta=resposta_texto, tempo_execucao=elapsed)
        sess.add(r)
        sess.commit()
        return resposta_texto

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_API_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{context}\n\nPergunta: {question}")
    ])
    chain = prompt | llm

    resp = chain.invoke({"context": context, "question": pergunta_texto})
    resposta_texto = resp.content if hasattr(resp, "content") else str(resp)

    elapsed = time.time() - start
    r = RespostaIA(pergunta_id=q.id, resposta=resposta_texto, tempo_execucao=elapsed)
    sess.add(r)
    sess.commit()
    return resposta_texto


# ------------------------------------------------------------
# Upload e processamento
# ------------------------------------------------------------

def upload_file(sess, filepath: str) -> int:
    with open(filepath, "rb") as f:
        data = f.read()
    nome = os.path.basename(filepath)
    tipo_nome = infer_tipo_from_name(nome)
    tipo = ensure_tipo(sess, tipo_nome)
    h = sha256_bytes(data)

    # Evitar duplicatas por hash
    existente = sess.query(Arquivo).filter_by(hash_sha256=h).one_or_none()
    if existente:
        sess.add(LogSistema(arquivo_id=existente.id, acao="upload_duplicado", detalhe=nome))
        sess.commit()
        print(f"[INFO] Arquivo já existe (ID={existente.id}). Pulando upload.")
        return existente.id

    # Criação do registro do arquivo
    arq = Arquivo(
        nome=nome,
        tipo_id=tipo.id,
        conteudo=data,
        tamanho=len(data),
        hash_sha256=h
    )
    sess.add(arq)
    sess.commit()

    # Extração do conteúdo
    try:
        texto = extract_content_by_type(tipo.nome, data)
    except Exception as e:
        texto = ""
        sess.add(LogSistema(arquivo_id=arq.id, acao="erro_extracao", detalhe=str(e)))
        sess.commit()
        print(f"[ERRO] Extração falhou: {e}")

    c = ConteudoExtraido(arquivo_id=arq.id, texto=texto)
    sess.add(c)
    sess.commit()

    # Indexação
    try:
        build_or_load_index_for_file(sess, c)
    except Exception as e:
        sess.add(LogSistema(arquivo_id=arq.id, acao="erro_indexacao", detalhe=str(e)))
        sess.commit()
        print(f"[ERRO] Indexação falhou: {e}")

    # Geração do resumo automático
    resumo_texto = ""
    try:
        if texto.strip():
            if GROQ_API_KEY:
                llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_API_MODEL)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Resuma o conteúdo a seguir de forma objetiva e em português claro."),
                    ("human", texto[:4000])  # limite de tokens para evitar truncamento
                ])
                chain = prompt | llm
                resp = chain.invoke({})
                resumo_texto = resp.content if hasattr(resp, "content") else str(resp)
            else:
                resumo_texto = texto[:500] + "..." if len(texto) > 500 else texto
        else:
            resumo_texto = "[Sem conteúdo extraído para resumir]"
    except Exception as e:
        resumo_texto = f"[Erro ao gerar resumo: {e}]"
        sess.add(LogSistema(arquivo_id=arq.id, acao="erro_resumo", detalhe=str(e)))
        sess.commit()
        print(f"[ERRO] Resumo falhou: {e}")

    # Salvar resumo na tabela
    r = Resumo(arquivo_id=arq.id, texto=resumo_texto)
    sess.add(r)
    sess.add(LogSistema(arquivo_id=arq.id, acao="resumo_gerado", detalhe=f"{len(resumo_texto)} caracteres"))
    sess.add(LogSistema(arquivo_id=arq.id, acao="upload_ok", detalhe=nome))
    sess.commit()

    print(f"[OK] Upload, processamento e resumo concluídos. ID={arq.id}")
    return arq.id


# ------------------------------------------------------------
# Consultas + gráficos
# ------------------------------------------------------------

def run_and_plot(sess, sql: str, descricao: str, chart_type: str = "barras") -> int:
    # Executa select e gera gráfico PNG simples
    cons = ConsultaSQL(sql=sql, descricao=descricao)
    sess.add(cons)
    sess.commit()

    # Executar SQL via engine (somente SELECT por segurança)
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Apenas SELECT é permitido nesta função.")

    df = pd.read_sql(sql, con=engine)

    # Salvar snapshot de dados
    dados_json = json.loads(df.to_json(orient="records"))

    # Plot básico
    img_path = os.path.join(CHART_DIR, f"grafico_{cons.id}.png")
    plt.figure()

    # heurística: se tiver 2 colunas, assume x e y
    if df.shape[1] >= 2:
        x = df.iloc[:, 0].astype(str)
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0)
        if chart_type == "linhas":
            plt.plot(x, y)
        elif chart_type == "pizza":
            plt.pie(y, labels=x, autopct='%1.1f%%')
        else:  # barras
            plt.bar(x, y)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    else:
        # fallback: histograma do que der
        for col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.dropna()
            if not s.empty:
                plt.hist(s)
                break
        plt.tight_layout()

    plt.savefig(img_path)
    plt.close()

    g = Grafico(consulta_id=cons.id, tipo=chart_type, dados=dados_json, imagem_path=img_path)
    sess.add(g)
    sess.commit()

    print(f"[OK] Consulta {cons.id} executada. Gráfico salvo em {img_path}")
    return cons.id

# ------------------------------------------------------------
# Setup do banco
# ------------------------------------------------------------

def create_all():
    Base.metadata.create_all(engine)
    with SessionLocal() as sess:
        for nome in ["pdf", "docx", "xlsx", "csv", "txt"]:
            ensure_tipo(sess, nome)
    print("[OK] Tabelas criadas e tipos iniciais inseridos.")


def drop_all():
    Base.metadata.drop_all(engine)
    print("[OK] Todas as tabelas foram removidas.")

# ------------------------------------------------------------
# CLI simples
# ------------------------------------------------------------
MENU = """
============== MENU ==============
1) Criar tabelas
2) Remover tabelas
3) Upload de arquivo
4) Perguntar sobre um arquivo
5) Consultas e gráficos prontos (3 exemplos)
6) Rodar consulta SQL customizada
0) Sair
> """


def menu():
    while True:
        try:
            op = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTchau!")
            return

        if op == "1":
            create_all()
        elif op == "2":
            conf = input("Tem certeza? (digite SIM): ").strip().upper()
            if conf == "SIM":
                drop_all()
        elif op == "3":
            path = input("Caminho do arquivo: ").strip().strip('"')
            if not os.path.isfile(path):
                print("[ERRO] Arquivo não encontrado.")
                continue
            with SessionLocal() as sess:
                upload_file(sess, path)
        elif op == "4":
            try:
                arquivo_id = int(input("ID do arquivo: "))
            except ValueError:
                print("ID inválido")
                continue
            pergunta = input("Pergunta: ")
            with SessionLocal() as sess:
                try:
                    resp = answer_question(sess, arquivo_id, pergunta)
                    print("\n===== RESPOSTA IA =====\n" + resp + "\n=======================\n")
                except Exception as e:
                    print(f"[ERRO] {e}")
        elif op == "5":
            with SessionLocal() as sess:
                # Consulta 1: quantidade de arquivos por tipo
                sql1 = (
                    "SELECT t.nome AS tipo, COUNT(a.id) AS qtde "
                    "FROM tipo_arquivo t LEFT JOIN arquivo a ON a.tipo_id = t.id "
                    "GROUP BY t.nome ORDER BY qtde DESC"
                )
                run_and_plot(sess, sql1, "Quantidade de arquivos por tipo", chart_type="barras")

                # Consulta 2: número de perguntas por arquivo
                sql2 = (
                    "SELECT a.nome AS arquivo, COUNT(p.id) AS perguntas "
                    "FROM arquivo a LEFT JOIN pergunta p ON p.arquivo_id = a.id "
                    "GROUP BY a.nome ORDER BY perguntas DESC"
                )
                run_and_plot(sess, sql2, "Perguntas por arquivo", chart_type="barras")

                # Consulta 3: tempo médio de resposta por tipo de arquivo
                sql3 = (
                    "SELECT t.nome AS tipo, COALESCE(AVG(r.tempo_execucao),0) AS tempo_medio_s "
                    "FROM tipo_arquivo t "
                    "LEFT JOIN arquivo a ON a.tipo_id = t.id "
                    "LEFT JOIN pergunta p ON p.arquivo_id = a.id "
                    "LEFT JOIN resposta_ia r ON r.pergunta_id = p.id "
                    "GROUP BY t.nome ORDER BY tempo_medio_s DESC"
                )
                run_and_plot(sess, sql3, "Tempo médio de resposta por tipo", chart_type="linhas")
        elif op == "6":
            with SessionLocal() as sess:
                sql = input("SQL (apenas SELECT): ")
                try:
                    df = pd.read_sql_query(sql, con=engine)
                    filename = f"consulta_{int(time.time())}.xlsx"
                    path = os.path.join(RESULT_CONSULTA_DIR, filename)
                    df.to_excel(path, index=False)
                    print(f"[OK] Consulta salva em {path}")
                except Exception as e:
                    print(f"[ERRO] {e}")
        elif op == "0":
            print("Até mais!")
            break
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    menu()
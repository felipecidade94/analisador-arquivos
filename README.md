# Analisador de Arquivos com IA e Interface Gráfica

O **Analisador de Arquivos** é uma aplicação desenvolvida em **Python**, que utiliza **Inteligência Artificial**, **LangChain**, **Groq** e **SQLAlchemy** para analisar, resumir e responder perguntas sobre diversos tipos de documentos.  
O sistema inclui uma **interface gráfica (Tkinter)** moderna e funcional, permitindo o envio de arquivos, execução de consultas SQL, geração de gráficos e interação direta com um modelo de linguagem.

---

## Funcionalidades Principais

- **Upload e processamento automático de arquivos**  
  Ao enviar um arquivo, o sistema:
  1. Extrai o conteúdo textual (PDF, DOCX, XLSX, CSV, TXT ou Markdown);  
  2. Gera embeddings e cria um índice FAISS para consultas semânticas;  
  3. Produz automaticamente um resumo via Groq (LangChain);  
  4. Registra todas as etapas e erros em logs detalhados.  

- **Consultas e respostas inteligentes (RAG)**  
  Permite fazer perguntas sobre o conteúdo dos arquivos, obtendo respostas baseadas em trechos contextualmente relevantes.  

- **Consultas SQL e gráficos**  
  - Executa consultas SQL personalizadas direto da interface;  
  - Gera gráficos automáticos com Matplotlib (quantidade de arquivos, perguntas por documento, tempo médio de resposta, etc.);  
  - Exporta resultados de consultas para planilhas Excel.

- **Interface gráfica interativa (Tkinter)**  
  Inclui uma área de chat com mensagens coloridas, botões laterais para ações rápidas, campo de envio de perguntas e exibição em tempo real das respostas da IA.

---

## Estrutura de Classes (SQLAlchemy ORM)

O sistema segue uma arquitetura orientada a objetos, com 10 entidades principais modeladas via SQLAlchemy.  
O diagrama abaixo mostra as classes e seus relacionamentos:

```mermaid
classDiagram
    class TipoArquivo {
        +int id
        +string nome
        +List~Arquivo~ arquivos
    }

    class Arquivo {
        +int id
        +string nome
        +int tipo_id
        +datetime data_upload
        +bytes conteudo
        +int tamanho
        +string hash_sha256
        +TipoArquivo tipo
        +ConteudoExtraido conteudo_extraido
        +List~Pergunta~ perguntas
        +Resumo resumo
        +List~Log~ logs
    }

    class ConteudoExtraido {
        +int id
        +int arquivo_id
        +text texto
        +Arquivo arquivo
        +List~Embedding~ embeddings
    }

    class Embedding {
        +int id
        +int conteudo_id
        +int num_chunks
        +int dim
        +string index_path
        +ConteudoExtraido conteudo
    }

    class Pergunta {
        +int id
        +int arquivo_id
        +text texto
        +datetime data
        +Arquivo arquivo
        +RespostaIA resposta
    }

    class RespostaIA {
        +int id
        +int pergunta_id
        +text resposta
        +float tempo_execucao
        +int tokens_input
        +int tokens_output
        +Pergunta pergunta
    }

    class Log {
        +int id
        +int arquivo_id
        +string acao
        +text detalhe
        +datetime data
        +Arquivo arquivo
    }

    class Resumo {
        +int id
        +int arquivo_id
        +text texto
        +datetime data
        +Arquivo arquivo
    }

    class ConsultaSQL {
        +int id
        +text sql
        +text descricao
        +datetime data
        +ResultadoConsulta resultado
    }

    class ResultadoConsulta {
        +int id
        +int consulta_id
        +string caminho_arquivo
        +json dados_json
        +int linhas
        +int colunas
        +datetime data_execucao
        +ConsultaSQL consulta
    }

    TipoArquivo "1" --> "n" Arquivo
    Arquivo "1" --> "1" ConteudoExtraido
    ConteudoExtraido "1" --> "n" Embedding
    Arquivo "1" --> "n" Pergunta
    Pergunta "1" --> "1" RespostaIA
    Arquivo "1" --> "1" Resumo
    Arquivo "1" --> "n" Log
    ConsultaSQL "1" --> "1" ResultadoConsulta
````

---

## Instalação e Execução

### 1. Criar e ativar o ambiente virtual (venv)

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS
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

### 4. Executar o sistema no terminal

```bash
python main.py
```

### 5. Rodar a interface gráfica

```bash
python interface.py
```

---

## Boas Práticas

* **Use ambientes virtuais (venv)** para isolar dependências e evitar conflitos de versão.
* **Não versione o arquivo `.env`** — ele deve conter credenciais locais.
* Prefira usar o ORM (**SQLAlchemy**) em vez de SQL cru, por segurança e portabilidade.
* Mantenha uma separação clara entre **lógica de aplicação (main.py)** e **interface (interface.py)**.
* Crie **commits curtos e descritivos**, e mantenha o README atualizado a cada nova versão.
* Teste uploads de diferentes formatos de arquivo antes de atualizar o repositório.

---

## Estrutura do Projeto

```
analisador-arquivos/
├── main.py                # Núcleo do sistema (banco, IA, extração, gráficos)
├── interface.py           # Interface Tkinter
├── requirements.txt       # Dependências do projeto
├── .env                   # Configurações de ambiente
├── charts/                # Gráficos gerados automaticamente
├── consultas/             # Consultas SQL salvas em Excel
├── indices_faiss/         # Índices vetoriais (FAISS)
├── tests/                 # Recursos de teste (logos, arquivos exemplo)
└── README.md              # Este documento
```

---

## Licença

Este projeto é distribuído sob a licença **MIT**.
Você pode usar, modificar e redistribuir livremente, desde que mantenha os créditos originais.

---

## Observação Final

O projeto segue princípios de **modularidade, reprodutibilidade e escalabilidade**, com uma arquitetura que combina **processamento local** (FAISS e SQLAlchemy) com **inteligência artificial baseada em LangChain**.
A interface gráfica transforma a complexidade técnica em uma experiência acessível e intuitiva, adequada tanto para demonstrações acadêmicas quanto para aplicações reais.

---

```

---

Quer que eu acrescente também uma seção **“🚀 Extensões futuras”**, listando ideias como adicionar `AnaliseArquivo`, suporte a busca global entre documentos ou exportação em PDF dos resumos? Isso deixa o README mais completo e com visão de evolução.
```

---

## Créditos

Desenvolvido por **Felipe Cidade Soares**

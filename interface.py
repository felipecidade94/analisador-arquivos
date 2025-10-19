import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import pandas as pd
import time
import main as m
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine

engine = create_engine(os.getenv('DATABASE_URL'))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---------------------------------------------
# CONFIGURAÇÃO DA JANELA
# ---------------------------------------------
janela = tk.Tk()
janela.title('ANALISADOR DE ARQUIVOS')
janela.geometry('980x600')
janela.configure(bg='#f5f6f7')
janela.resizable(False, True)

# ---------------------------------------------
# ESTILOS
# ---------------------------------------------
style = ttk.Style()
style.theme_use('clam')

style.configure(
    'Custom.TButton',
    font=('Segoe UI', 11, 'bold'),
    padding=6,
    background='#e0f0ff',
    foreground='#222',
)
style.map('Custom.TButton',
        background=[('active', '#c2e1ff')])

style.configure('Custom.TLabel', font=('Segoe UI', 18, 'bold'), background='#f5f6f7')

# ---------------------------------------------
# FRAMES
# ---------------------------------------------
frame_chat = tk.Frame(janela, bg='white', bd=1, relief='solid')
frame_chat.pack(side='left', fill='both', expand=True, padx=(20, 10), pady=20)

frame_menu = tk.Frame(janela, bg='#f5f6f7')
frame_menu.pack(side='right', fill='y', padx=(0, 20), pady=20)

# ---------------------------------------------
# LOGO
# ---------------------------------------------
try:
    image_path = './tests/logo.png'  # Ajuste conforme teu projeto
    img = Image.open(image_path)
    img = img.resize((190, 160))
    img_tk = ImageTk.PhotoImage(img)
    lbl_logo = ttk.Label(frame_menu, image=img_tk, background='#f5f6f7')
    lbl_logo.image = img_tk
    lbl_logo.pack(pady=(10, 15))
except Exception:
    lbl_logo = ttk.Label(frame_menu, text='LOGO', style='Custom.TLabel')
    lbl_logo.pack(pady=(10, 15))

# ---------------------------------------------
# ÁREA DE TEXTO / CHAT
# ---------------------------------------------
frame_canvas = tk.Frame(frame_chat)
frame_canvas.pack(fill='both', expand=True, padx=10, pady=10)

canvas = tk.Canvas(frame_canvas, bg='white', highlightthickness=0)
scrollbar = ttk.Scrollbar(frame_canvas, orient='vertical', command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side='right', fill='y')
canvas.pack(side='left', fill='both', expand=True)

frame_mensagens = tk.Frame(canvas, bg='white')
canvas.create_window((0, 0), window=frame_mensagens, anchor='nw')

def atualizar_scroll(event=None):
    canvas.configure(scrollregion=canvas.bbox('all'))


frame_mensagens.bind('<Configure>', atualizar_scroll)

def exibir_mensagem(remetente, texto):
    cor = '#e0ffe0' if remetente == 'Você' else '#d7ebf9'

    # Define a largura de quebra com base no tamanho atual do frame
    largura_msg = frame_chat.winfo_width() - 120  # margem lateral

    # Garante que a largura esteja válida (se a janela acabou de abrir)
    if largura_msg < 300:
        largura_msg = 600

    msg = tk.Message(
        frame_mensagens,
        text=f'{remetente}: {texto}',
        bg=cor,
        width=largura_msg,
        justify='left',
        anchor='w',
        padx=10,
        pady=5,
        font=('Segoe UI', 10)
    )
    msg.pack(fill='x', padx=10, pady=5, anchor='w')

    # Atualiza a região rolável
    frame_mensagens.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox('all'))
    canvas.yview_moveto(1.0)


exibir_mensagem('Sistema', '[OK] Conectado ao banco com sucesso!')
# ---------------------------------------------
# FUNÇÕES DE BOTÕES
# ---------------------------------------------
def criar_tabelas():
    msg = m.create_all()
    messagebox.showinfo('Sucesso', msg)
    exibir_mensagem('Sistema', msg)

def remover_tabelas():
    if messagebox.askyesno('Confirmação', 'Tem certeza que deseja remover todas as tabelas?'):
        msg = m.drop_all()
        exibir_mensagem('Sistema', msg)

def upload_arquivo():
    caminho = filedialog.askopenfilename(
        title='Selecione um arquivo',
        filetypes=[
            ('Todos os suportados', '*.pdf; *.docx; *.xlsx; *.xls; *.csv; *.txt; *.md'),
            ('PDF', '*.pdf'),
            ('DOCX', '*.docx'),
            ('Excel', '*.xlsx;*.xls'),
            ('CSV', '*.csv'),
            ('TXT', '*.txt'),
            ('Markdown', '*.md')
        ]
    )
    if not caminho:
        return

    try:
        with m.SessionLocal() as sess:
            arq_id = m.upload_file(sess, caminho)

            # Busca todos os logs referentes a esse arquivo
            logs = (
                sess.query(m.Log)
                .filter(m.Log.arquivo_id == arq_id)
                .order_by(m.Log.data.asc())
                .all()
            )

            # Mostra a mensagem principal
            exibir_mensagem('Sistema', f'Upload concluído. ID do arquivo: {arq_id}')

            # Mostra cada evento registrado no log
            for log in logs:
                cor = '#ffebcc' if 'erro' in (log.acao or '').lower() else '#d6f5d6'
                exibir_mensagem(
                    'Log',
                    f"[{log.data.strftime('%H:%M:%S')}] {log.acao.upper()} → {log.detalhe}",
                )

    except Exception as e:
        exibir_mensagem('Erro', f'Ocorreu um erro ao enviar o arquivo: {e}')


# ---------------------------------------------
# VARIÁVEIS DE CONTROLE
# ---------------------------------------------

id_arquivo_escolhido = None

# ---------------------------------------------
# FUNÇÃO PARA SELECIONAR O ARQUIVO
# ---------------------------------------------
def perguntar_arquivo():
    global id_arquivo_escolhido
    janela_id = tk.Toplevel(janela)
    janela_id.title('Selecionar arquivo para perguntas')
    janela_id.geometry('380x160')
    sql = 'SELECT a.id, a.nome FROM arquivo a'
    df = pd.read_sql_query(sql, engine)
    lista_arquivos = df['nome'].to_list()
    ttk.Label(janela_id, text='Escolha o arquivo:').pack(pady=10)
    combo_aquivo = ttk.Combobox(janela_id, width=80, values=lista_arquivos)
    combo_aquivo.pack(pady=5)
    
    def confirmar_id():
        global id_arquivo_escolhido
        valor = combo_aquivo.get().strip()
        id_arquivo_escolhido = lista_arquivos.index(valor)+1
        if not valor:
            messagebox.showwarning('Aviso', 'É necessário escolher um arquivo.')
            return
        exibir_mensagem('Sistema', f'Arquivo {lista_arquivos[id_arquivo_escolhido-1]} selecionado para perguntas.')
        janela_id.destroy()

    ttk.Button(janela_id, text='Confirmar', style='Custom.TButton', command=confirmar_id).pack(pady=10)


# ---------------------------------------------
# FUNÇÃO PARA ENVIAR PERGUNTA VIA ENTRY
# ---------------------------------------------
def enviar_pergunta():
    global id_arquivo_escolhido
    pergunta = entry_enviar.get().strip()
    if not pergunta:
        return
    if id_arquivo_escolhido is None:
        messagebox.showwarning('Aviso', 'Você precisa selecionar o ID do arquivo antes de perguntar.')
        return

    exibir_mensagem('Você', pergunta)
    entry_enviar.delete(0, tk.END)

    janela.update_idletasks()

    with m.SessionLocal() as sess:
        try:
            resposta = m.answer_question(sess, int(id_arquivo_escolhido), pergunta)
            exibir_mensagem('IA', resposta)
        except Exception as e:
            exibir_mensagem('Sistema', f'[ERRO] {e}')
            messagebox.showerror('Erro', str(e))


def consultas_prontas():
    try:
        exibir_mensagem('Sistema', 'Executando consultas e gráficos prontos...')
        with m.SessionLocal() as sess:
            # Consulta 1
            sql1 = '''
                SELECT t.nome AS tipo, COUNT(a.id) AS qtde 
                FROM tipo_arquivo t LEFT JOIN arquivo a ON a.tipo_id = t.id 
                GROUP BY t.nome ORDER BY qtde DESC
            '''
            m.run_and_plot('Quantidade de arquivos por tipo', 'Tipo', 'Quantidade', sql1, 'Arquivos por tipo', 'barras')

            # Consulta 2
            sql2 = '''
                SELECT a.nome AS arquivo, COUNT(p.id) AS perguntas 
                FROM arquivo a LEFT JOIN pergunta p ON p.arquivo_id = a.id 
                GROUP BY a.nome ORDER BY perguntas DESC
            '''
            m.run_and_plot('Número de perguntas por arquivo', 'Arquivo', 'Perguntas', sql2, 'Perguntas por arquivo', 'barras')

            # Consulta 3
            sql3 = '''
                SELECT t.nome AS tipo, COALESCE(AVG(r.tempo_execucao),0) AS tempo_medio_s 
                FROM tipo_arquivo t 
                LEFT JOIN arquivo a ON a.tipo_id = t.id 
                LEFT JOIN pergunta p ON p.arquivo_id = a.id 
                LEFT JOIN resposta_ia r ON r.pergunta_id = p.id 
                GROUP BY t.nome ORDER BY tempo_medio_s DESC
            '''
            m.run_and_plot('Tempo médio de resposta por tipo', 'Tipo', 'Tempo (s)', sql3, 'Tempo médio', 'linhas')

        exibir_mensagem('Sistema', 'Consultas e gráficos prontos concluídos!')
        messagebox.showinfo('Sucesso', 'Consultas e gráficos gerados com sucesso!')
    except Exception as e:
        messagebox.showerror('Erro', f'Falha ao gerar gráficos: {e}')


def consulta_sql():
    janela_sql = tk.Toplevel(janela)
    janela_sql.title('Consulta SQL customizada')
    janela_sql.geometry('500x400')

    ttk.Label(janela_sql, text='Digite uma consulta SELECT:').pack(pady=10)
    entry_sql = tk.Text(janela_sql, height=18, width=220)
    entry_sql.pack(padx=10)

    def executar_sql():
        sql = entry_sql.get('1.0', 'end').strip()
        if not sql.lower().startswith('select'):
            messagebox.showwarning('Erro', 'Apenas consultas SELECT são permitidas.')
            return
        with m.SessionLocal() as sess:
            try:
                cons = m.ConsultaSQL(sql=sql, descricao='Consulta via interface')
                sess.add(cons)
                sess.commit()
                df = pd.read_sql_query(sql, con=m.engine)
                timestamp = int(time.time())
                path = f'./consultas/consulta_custom_{cons.id}_{timestamp}.xlsx'
                df.to_excel(path, index=False)
                exibir_mensagem('Sistema', f'Consulta salva em {path}')
                messagebox.showinfo('Sucesso', f'Consulta salva em {path}')
            except Exception as e:
                messagebox.showerror('Erro', str(e))

    ttk.Button(janela_sql, text='Executar', style='Custom.TButton', command=executar_sql).pack(pady=15)

# ---------------------------------------------
# BOTÕES DO MENU
# ---------------------------------------------
botoes = {
    'Criar tabelas': criar_tabelas,
    'Remover tabelas': remover_tabelas,
    'Upload de arquivo': upload_arquivo,
    'Perguntar sobre um arquivo': perguntar_arquivo,
    '3 Gráficos prontos': consultas_prontas,
    'Consulta SQL customizada': consulta_sql,
    'Sair': janela.destroy
}

for nome, funcao in botoes.items():
    ttk.Button(frame_menu, text=nome, style='Custom.TButton', width=30, command=funcao).pack(pady=5, padx=10)

# ---------------------------------------------
# CAMPO DE ENTRADA / ENVIO
# ---------------------------------------------
frame_entry = tk.Frame(frame_chat, bg='white')
frame_entry.pack(fill='x', padx=10, pady=10)
entry_enviar = ttk.Entry(frame_entry, font=('Segoe UI', 11))
entry_enviar.pack(side='left', fill='x', expand=True, padx=(0, 5))
ttk.Button(frame_entry, text='Enviar', style='Custom.TButton', command=enviar_pergunta).pack(side='right')

# ---------------------------------------------
# EXECUÇÃO
# ---------------------------------------------
janela.mainloop()
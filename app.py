# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import pandas as pd
import time
import main as m

# ============================================================
# JANELA PRINCIPAL
# ============================================================
janela = tk.Tk()
janela.title('ANALISADOR DE ARQUIVOS')
janela.geometry('1200x700')
janela.configure(bg='#f5f6f7')
janela.state('zoomed')
janela.resizable(True, True)

# ============================================================
# ESTILOS
# ============================================================
style = ttk.Style()
style.theme_use('clam')

style.configure(
    'Custom.TButton',
    font=('Segoe UI', 11, 'bold'),
    padding=6,
    background='#e0f0ff',
    foreground='#222',
)
style.map('Custom.TButton', background=[('active', '#c2e1ff')])
style.configure('Custom.TLabel', font=('Segoe UI', 18, 'bold'), background='#f5f6f7')

# ============================================================
# LAYOUT
# ============================================================
frame_chat = tk.Frame(janela, bg='white', bd=1, relief='solid')
frame_chat.pack(side='left', fill='both', expand=True, padx=(20, 10), pady=20)

frame_menu = tk.Frame(janela, bg='#f5f6f7')
frame_menu.pack(side='right', fill='y', padx=(0, 20), pady=20)

# ============================================================
# LOGO
# ============================================================
try:
    image_path = './img/logo.png'
    img = Image.open(image_path).resize((190, 160))
    img_tk = ImageTk.PhotoImage(img)
    lbl_logo = ttk.Label(frame_menu, image=img_tk, background='#f5f6f7')
    lbl_logo.image = img_tk
    lbl_logo.pack(pady=(10, 15))
except Exception:
    ttk.Label(frame_menu, text='ANALISADOR DE ARQUIVOS', style='Custom.TLabel').pack(pady=(10, 15))

# ============================================================
# ÁREA DE MENSAGENS
# ============================================================
frame_canvas = tk.Frame(frame_chat)
frame_canvas.pack(fill='both', expand=True, padx=10, pady=10)

canvas = tk.Canvas(frame_canvas, bg='white', highlightthickness=0)
scrollbar = ttk.Scrollbar(frame_canvas, orient='vertical', command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side='right', fill='y')
canvas.pack(side='left', fill='both', expand=True)

frame_mensagens = tk.Frame(canvas, bg='white')
canvas.create_window((0, 0), window=frame_mensagens, anchor='nw')

caminhos = ['charts', 'consultas', 'indices_faiss']

def atualizar_scroll(event=None):
    canvas.configure(scrollregion=canvas.bbox('all'))
frame_mensagens.bind('<Configure>', atualizar_scroll)

def exibir_mensagem(remetente, texto, cor=None):
    cor = cor or ('#e0ffe0' if remetente == 'Você' else '#d7ebf9')
    largura_msg = max(frame_chat.winfo_width() - 120, 600)
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
    frame_mensagens.update_idletasks()
    canvas.yview_moveto(1.0)

exibir_mensagem('Sistema', '[OK] Interface inicializada com sucesso!')

# ============================================================
# FUNÇÕES
# ============================================================
def criar_tabelas():
    try:
        msg = m.create_all()
        exibir_mensagem('Sistema', msg)
        messagebox.showinfo('Sucesso', msg)
    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))

def remover_tabelas():
    if not m.verificar_banco():
        msg = 'Não é possível remover as tabelas, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Sistema', msg)
        return messagebox.showerror('ERRO', msg)
    conf = messagebox.askyesno('Confirmação', 'Tem certeza que deseja remover todas as tabelas e pastas?')
    if not conf:
        return messagebox.showerror('ERRO', 'Ação cancelada pelo usuário.')
    janela.bell()
    aviso = messagebox.askyesno('AVISO', 'Essa ação é irreversível! Deseja continuar?')
    if not aviso:
        return messagebox.showerror('ERRO', 'Ação cancelada pelo usuário.')
    try:
        msg = m.drop_all()
        exibir_mensagem('Sistema', msg)
        messagebox.showinfo('INFO', msg)
    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))
    

def upload_arquivo():    # Função para mandar um mensagem de ERRO caso o banco não exista
    if not m.verificar_banco():
        msg = 'Não é possível fazer upload de arquivos, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return

    caminho = filedialog.askopenfilename(
        title='Selecione um arquivo',
        filetypes=[
            ('Todos os suportados', '*.pdf; *.docx; *.xlsx; *.xls; *.csv; *.txt; *.md'),
            ('PDF', '*.pdf'),
            ('Word', '*.docx'),
            ('Excel', '*.xlsx;*.xls'),
            ('CSV', '*.csv'),
            ('TXT', '*.txt'),
            ('Markdown', '*.md'),
        ]
    )
    if not caminho:
        return

    try:
        with m.SessionLocal() as sess:
            resultado = m.upload_file(sess, caminho)

        arq_id = resultado['id']

        if resultado['duplicado']:
            exibir_mensagem('Sistema', f'O arquivo já existe no banco. ID={arq_id}. Upload pulado')
        else:
            exibir_mensagem('Sistema', f'Upload concluído com sucesso. ID={arq_id}')
            time.sleep(0.5)

                # Exibir apenas o último log do arquivo
        df_logs = pd.read_sql_query(
            f'''
            SELECT acao, detalhe, data 
            FROM log 
            WHERE arquivo_id={arq_id} 
            ORDER BY data DESC 
            LIMIT 1
            ''',
            con=m.engine
        )

        if not df_logs.empty:
            log = df_logs.iloc[0]
            cor = '#ffe6e6' if 'erro' in log['acao'].lower() else '#e6ffe6'
            if 'duplicado' in log['acao'].lower():
                cor = '#fff8dc'  # amarelo claro para duplicado
            hora = pd.to_datetime(log['data']).strftime('%d-%m-%y-%H:%M:%S')
            exibir_mensagem('Log', f"[{hora}] {log['acao']} → {log['detalhe']}", cor)

    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))

id_arquivo_escolhido = None

def perguntar_arquivo():
    if not m.verificar_banco():
        msg = 'Não é possível perguntar sobre arquivos, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return
    
    
    global id_arquivo_escolhido
    janela_id = tk.Toplevel(janela)
    janela_id.title('Selecionar arquivo para perguntas')
    janela_id.geometry('420x160')
    janela_id.transient(janela)
    janela_id.grab_set()

    try:
        df = pd.read_sql_query("SELECT id, nome FROM arquivo ORDER BY id DESC", con=m.engine)
    except Exception as e:
        exibir_mensagem('Erro', f'Erro ao listar arquivos: {e}')
        messagebox.showerror('Erro', str(e))
        return

    if df.empty:
        messagebox.showerror('ERRO', 'Nenhum arquivo encontrado. Faça upload primeiro.')
        janela_id.destroy()
        return

    ttk.Label(janela_id, text='Escolha o arquivo:').pack(pady=10)
    combo = ttk.Combobox(janela_id, width=50, values=df['nome'].tolist(), state='readonly')
    combo.pack(pady=5)

    def confirmar():
        global id_arquivo_escolhido
        nome = combo.get().strip()
        if not nome:
            messagebox.showwarning('Aviso', 'Selecione um arquivo.')
            return
        try:
            id_arquivo_escolhido = int(df.loc[df['nome'] == nome, 'id'].iloc[0])
        except Exception as e:
            messagebox.showerror('Erro', f'Seleção inválida: {e}')
            return
        exibir_mensagem('Sistema', f'Arquivo "{nome}" selecionado (ID={id_arquivo_escolhido}) para perguntas.')
        janela_id.destroy()

    ttk.Button(janela_id, text='Confirmar', style='Custom.TButton', command=confirmar).pack(pady=10)

def enviar_pergunta():
    global id_arquivo_escolhido
    pergunta = entry_enviar.get().strip()
    if not pergunta:
        return
    if id_arquivo_escolhido is None:
        messagebox.showwarning('Aviso', 'Selecione um arquivo antes de perguntar.')
        return
    exibir_mensagem('Você', pergunta)
    entry_enviar.delete(0, tk.END)
    try:
        with m.SessionLocal() as sess:
            resposta = m.answer_question(sess, id_arquivo_escolhido, pergunta)
        exibir_mensagem('IA', resposta)
    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))

def graficos_prontos():
    if not m.verificar_banco():
        msg = 'Não é possível gerar os gráficos prontos, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return
    
    try:
        df = pd.read_sql_query('SELECT id, nome FROM arquivo ORDER BY id DESC', con=m.engine)
    except Exception as e:
        exibir_mensagem('Erro', f'Erro ao gerar os gráficos: {e}')
        messagebox.showerror('Erro', str(e))
        return

    if df.empty:
        messagebox.showerror('Aviso', 'Nenhum arquivo encontrado. Faça upload primeiro.')
        return

    exibir_mensagem('Sistema', 'Executando consultas e gráficos prontos...')
    try:
        # Usa diretamente as funções do main.py
        sqls = [
            (
                "Quantidade de arquivos x tipo",
                "Tipo",
                "Quantidade",
                '''
                SELECT t.nome AS tipo, COUNT(a.id) AS qtde
                FROM tipo_arquivo t
                LEFT JOIN arquivo a ON a.tipo_id = t.id
                GROUP BY t.nome ORDER BY qtde DESC
                ''',
                "Arquivos por tipo",
                "barras"
            ),
            (
                "Perguntas x arquivo",
                "Arquivo",
                "Nº perguntas",
                '''
                SELECT a.nome AS arquivo, COUNT(p.id) AS perguntas
                FROM arquivo a
                LEFT JOIN pergunta p ON p.arquivo_id = a.id
                GROUP BY a.nome ORDER BY perguntas DESC
                ''',
                "Perguntas por arquivo",
                "barras"
            ),
            (
                "Tempo médio de resposta x tipo",
                "Tipo",
                "Tempo (s)",
                '''
                SELECT t.nome AS tipo, COALESCE(AVG(r.tempo_execucao),0) AS tempo_medio_s
                FROM tipo_arquivo t
                LEFT JOIN arquivo a ON a.tipo_id = t.id
                LEFT JOIN pergunta p ON p.arquivo_id = a.id
                LEFT JOIN resposta_ia r ON r.pergunta_id = p.id
                GROUP BY t.nome ORDER BY tempo_medio_s DESC
                ''',
                "Tempo médio de resposta",
                "linhas"
            ),
        ]
        for args in sqls:
            m.run_and_plot(*args)
        exibir_mensagem('Sistema', 'Consultas e gráficos prontos concluídos.')
    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))

def consulta_sql():
    if not m.verificar_banco():
        msg = 'Não é possível executar consultas, porque as tabelas do banco estão vazias. Faça upload de novos arquivos para fazer fazer perguntas.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return
    
    janela_sql = tk.Toplevel(janela)
    janela_sql.title('Consulta SQL customizada')
    janela_sql.geometry('600x450')
    janela_sql.transient(janela)
    janela_sql.grab_set()

    try:
        df = pd.read_sql_query('''SELECT a.id as id,
                                a.nome as nome
                                FROM arquivo a
                                ORDER BY nome''', con=m.engine)
        if df.empty:
            messagebox.showerror('ERRO', 'Nenhum arquivo disponível para consulta.')
            janela_sql.destroy()
            return
    except Exception as e:
        messagebox.showerror('Erro', f'Falha ao buscar arquivos: {e}')
        janela_sql.destroy()
        return
    
    ttk.Label(janela_sql, text='Digite uma consulta SELECT:').pack(pady=10)
    entry_sql = tk.Text(janela_sql, height=15, width=80, font=('Consolas', 10))
    entry_sql.pack(padx=10, fill='both', expand=True)

    def executar():
        sql = entry_sql.get('1.0', 'end').strip()
        if not sql.lower().startswith('select'):
            messagebox.showwarning('Erro', 'Apenas consultas SELECT são permitidas.')
            return
        try:
            df = pd.read_sql_query(sql, con=m.engine)
            if df.empty:
                exibir_mensagem('Sistema', 'Nenhum resultado retornado.')
                messagebox.showinfo('Aviso', 'Nenhum resultado retornado.')
                return
            path = os.path.join('consultas', f'consulta_{int(time.time())}.xlsx')
            df.to_excel(path, index=False)
            exibir_mensagem('Sistema', f'Consulta executada com sucesso. Resultado salvo em {path}')
            messagebox.showinfo('Sucesso', f'Resultado salvo em:\n{path}')
        except Exception as e:
            exibir_mensagem('Erro', str(e))
            messagebox.showerror('Erro', str(e))

    ttk.Button(janela_sql, text='Executar', style='Custom.TButton', command=executar).pack(pady=10)

def remover_arquivo():
    if not m.verificar_banco():
        msg = 'Não é possível remover arquivos, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return

    """Abre uma janela para remover um arquivo existente pelo ID."""
    janela_remover = tk.Toplevel(janela)
    janela_remover.title('Remover arquivo')
    janela_remover.geometry('380x200')

    try:
        df = pd.read_sql_query('''SELECT a.id as id,
                                a.nome as nome
                                FROM arquivo a
                                ORDER BY nome''', con=m.engine)
        if df.empty:
            messagebox.showerror('ERRO', 'Nenhum arquivo disponível para remoção.')
            janela_remover.destroy()
            return
    except Exception as e:
        messagebox.showerror('Erro', f'Falha ao buscar arquivos: {e}')
        janela_remover.destroy()
        return

    ttk.Label(janela_remover, text='Selecione o arquivo para remover:').pack(pady=10)
    combo = ttk.Combobox(janela_remover, width=60, values=df['nome'].to_list())
    combo.pack(pady=5)

    def confirmar_remocao():
        nome = combo.get()
        if not nome:
            messagebox.showwarning('Aviso', 'Selecione um arquivo.')
            return
        arq_id = int(df.loc[df['nome'] == nome, 'id'].iloc[0])
        conf = messagebox.askyesno('Confirmação', f'Tem certeza que deseja remover o arquivo "{nome}" (ID={arq_id})?')
        if conf:
            janela_remover.bell()
            aviso = messagebox.askyesno('Confirmação', 'Essa ação é irreversível! Deseja continuar?')
            if aviso:
                with m.SessionLocal() as sess:
                    resultado = m.remove_file(sess, arq_id)
                exibir_mensagem('Sistema', resultado)
                if '[OK]' in resultado:
                    messagebox.showinfo('Sucesso', resultado)
                else:
                    messagebox.showerror('Erro', resultado)
                janela_remover.destroy()
        return

    ttk.Button(janela_remover, text='Remover', style='Custom.TButton', command=confirmar_remocao).pack(pady=15)

def listar_arquivos():
    if not m.verificar_banco():
        msg = 'Não é possível listar os arquivos, porque o banco ainda não existe. Clique no botão "Criar tabelas" para criar o banco.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return

    if not m.verificar_tabelas():
        msg = 'Não é possível listar os arquivos, porque as tabelas estão vazias. Faça upload de novos arquivos primeiro.'
        exibir_mensagem('Erro', msg)
        messagebox.showerror('Erro', msg)
        return
    
    try:
        df_arquivos = pd.read_sql_query('''
            SELECT a.id, a.nome
            FROM arquivo a  
            ORDER BY a.id
        ''', m.engine)
        lista_arquivos_ids = df_arquivos['id'].tolist()
        lista_arquivos_nomes = df_arquivos['nome'].tolist()
        arquivos = '\n'.join(f'ID: {id} | Nome: {nome}' for id, nome in zip(lista_arquivos_ids, lista_arquivos_nomes))
        if df_arquivos.empty:
            messagebox.showerror('ERRO', 'Nenhum arquivo encontrado. Faça upload primeiro.')
            return
    except Exception as e:
        exibir_mensagem('Erro', str(e))
        messagebox.showerror('Erro', str(e))
        return
    
    messagebox.showinfo('Lista de Arquivos', arquivos)

def sair():
    sair = messagebox.askyesno('SAIR', 'Tem certeza que deseja sair?')
    if sair:
        janela.destroy()

# ============================================================
# BOTÕES DO MENU
# ============================================================
botoes = {
    'Criar tabelas': criar_tabelas,
    'Remover tabelas': remover_tabelas,
    'Remover arquivo': remover_arquivo,
    'Listar arquivos': listar_arquivos,
    'Upload de arquivo': upload_arquivo,
    'Perguntar sobre um arquivo': perguntar_arquivo,
    '3 Gráficos prontos': graficos_prontos,
    'Consulta SQL customizada': consulta_sql,
    'Sair': sair,
}

for nome, funcao in botoes.items():
    ttk.Button(frame_menu, text=nome, style='Custom.TButton', width=30, command=funcao).pack(pady=5, padx=10)

# ============================================================
# CAMPO DE ENTRADA INFERIOR
# ============================================================
frame_entry = tk.Frame(frame_chat, bg='white')
frame_entry.pack(fill='x', padx=10, pady=10)

entry_enviar = ttk.Entry(frame_entry, font=('Segoe UI', 11))
entry_enviar.pack(side='left', fill='x', expand=True, padx=(0, 5))

ttk.Button(frame_entry, text='Enviar', style='Custom.TButton', command=enviar_pergunta).pack(side='right')

janela.bind('<Return>', lambda e: enviar_pergunta())
janela.mainloop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para a função remove_file_by_id
"""
import main as m
import os

def test_remove_file():
    """Testa a função de remoção de arquivo"""
    print("=== TESTE DA FUNÇÃO REMOVE_FILE_BY_ID ===\n")
    
    # Criar tabelas se não existirem
    print("1. Criando tabelas...")
    print(m.create_all())
    
    # Listar arquivos existentes
    print("\n2. Listando arquivos existentes...")
    try:
        with m.SessionLocal() as sess:
            arquivos = sess.query(m.Arquivo).all()
            if not arquivos:
                print("Nenhum arquivo encontrado. Teste concluído.")
                return
            
            print(f"Arquivos encontrados: {len(arquivos)}")
            for arq in arquivos:
                print(f"  - ID: {arq.id}, Nome: {arq.nome}")
            
            # Testar remoção do primeiro arquivo
            primeiro_id = arquivos[0].id
            print(f"\n3. Testando remoção do arquivo ID {primeiro_id}...")
            resultado = m.remove_file_by_id(sess, primeiro_id)
            print(f"Resultado: {resultado}")
            
            # Verificar se foi removido
            print("\n4. Verificando se o arquivo foi removido...")
            arquivo_removido = sess.get(m.Arquivo, primeiro_id)
            if arquivo_removido is None:
                print("✓ Arquivo removido com sucesso!")
            else:
                print("✗ Arquivo ainda existe!")
            
            # Testar remoção de arquivo inexistente
            print("\n5. Testando remoção de arquivo inexistente...")
            resultado_inexistente = m.remove_file_by_id(sess, 99999)
            print(f"Resultado: {resultado_inexistente}")
            
    except Exception as e:
        print(f"Erro durante o teste: {e}")
    
    print("\n=== TESTE CONCLUÍDO ===")

if __name__ == "__main__":
    test_remove_file()

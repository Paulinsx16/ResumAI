#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para buscar e resumir notícias do Brasil
Utiliza APIs de notícias e processamento de linguagem natural (NLP)
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv

# Importar bibliotecas de NLP
try:
    from transformers import pipeline
except ImportError:
    print("Instale as dependências: pip install transformers torch sentencepiece")

load_dotenv()

class NewsAggregator:
    """Classe para agregar e resumir notícias do Brasil"""
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.summarizer = None
        self._init_summarizer()
    
    def _init_summarizer(self):
        """Inicializa o pipeline de sumarização com modelo em Português"""
        try:
            # Substituído por um modelo ajustado para Português (PT-BR)
            model_name = "phillipe-carvalho/ptt5-base-portuguese-vocab-sum-newswire-portuguese"
            self.summarizer = pipeline("summarization", model=model_name)
            print("✓ Sumarizador (PT-BR) carregado com sucesso")
        except Exception as e:
            print(f"⚠ Erro ao carregar sumarizador (usando fallback de corte de texto): {e}")
    
    def fetch_news_newsapi(self, query: str = "Brasil", days: int = 1) -> List[Dict]:
        """Busca notícias usando NewsAPI e padroniza a resposta"""
        if not self.newsapi_key:
            print("⚠ NEWSAPI_KEY não configurada no arquivo .env")
            return []
        
        params = {
            'q': query,
            'language': 'pt',
            'sortBy': 'publishedAt',
            'apiKey': self.newsapi_key,
            'from': (datetime.now() - timedelta(days=days)).date().isoformat(),
            'pageSize': 5 # Reduzido para testar mais rápido
        }
        
        try:
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                normalized_articles = []
                for art in data.get('articles', []):
                    # Padroniza a extração da fonte para evitar erros de dicionário vs string
                    source_name = art.get('source', {}).get('name', 'NewsAPI') if isinstance(art.get('source'), dict) else 'NewsAPI'
                    normalized_articles.append({
                        'title': art.get('title'),
                        'description': art.get('description', ''),
                        'content': art.get('content', ''),
                        'url': art.get('url'),
                        'source': source_name,
                        'publishedAt': art.get('publishedAt')
                    })
                print(f"✓ {len(normalized_articles)} notícias encontradas via NewsAPI")
                return normalized_articles
            else:
                print(f"⚠ Erro na API: {data.get('message', 'Desconhecido')}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"✗ Erro ao buscar notícias na NewsAPI: {e}")
            return []
    
    def fetch_news_g1(self) -> List[Dict]:
        """Busca notícias principais diretamente do feed do G1 via BeautifulSoup"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("⚠ BeautifulSoup não instalado. Instale com: pip install beautifulsoup4")
            return []
        
        try:
            url = "https://g1.globo.com"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Seletores atualizados para focar nas classes de chamadas de matérias do G1
            for item in soup.find_all('a', class_='feed-post-link')[:5]:
                title = item.get_text(strip=True)
                link = item.get('href', '')
                
                if title and len(title) > 15:
                    articles.append({
                        'title': title,
                        'description': title, # G1 na home não entrega a descrição facilmente, usamos o título como base
                        'content': '',
                        'url': link,
                        'source': 'G1',
                        'publishedAt': datetime.now().isoformat()
                    })
            
            print(f"✓ {len(articles)} notícias encontradas via G1")
            return articles
        except Exception as e:
            print(f"⚠ Erro ao buscar notícias do G1: {e}")
            return []
    
    def summarize_text(self, text: str, max_length: int = 80, min_length: int = 30) -> str:
        """Resume um texto em português usando o modelo de IA"""
        if not self.summarizer:
            return text[:150] + "..."
        
        # O PTT5 exige que o texto não seja extremamente curto para resumir
        if len(text.split()) < 15:
            return text
        
        try:
            # Adiciona o prefixo esperado por alguns modelos T5 (opcional, mas ajuda na precisão)
            inputs = f"summarize: {text}"
            summary = self.summarizer(inputs, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback seguro caso o texto estoure o limite de tokens do modelo (512 tokens)
            return text[:150] + "..."
    
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Processa e resume artigos padronizados"""
        processed = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Sem título')
            print(f"[{i}/{len(articles)}] Processando: {title[:50]}...")
            
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Filtra sujeiras comuns da NewsAPI (ex: contagem de caracteres ao final)
            if content and "chars]" in content:
                content = content.split("[+")[0]
                
            text_to_summarize = description if len(description) > len(content) else content
            if not text_to_summarize:
                text_to_summarize = title
            
            summary = self.summarize_text(text_to_summarize)
            
            processed.append({
                'titulo': title,
                'fonte': article.get('source', 'Desconhecida'),
                'url': article.get('url', ''),
                'data': article.get('publishedAt', ''),
                'resumo': summary
            })
        
        return processed
    
    def generate_report(self, articles: List[Dict]) -> str:
        """Gera relatório em texto formatado"""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║          RESUMO DE NOTÍCIAS DO BRASIL                          ║
║          Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}                  ║
╚════════════════════════════════════════════════════════════════╝

Total de notícias processadas: {len(articles)}
"""
        for i, article in enumerate(articles, 1):
            report += f"""
{'─' * 60}
📰 [{i}] {article['titulo']}
{'─' * 60}
Fonte: {article['fonte']}
Data:  {article['data'][:10] if article['data'] else 'N/A'}

Resumo:
{article['resumo']}

Acesse: {article['url']}
"""
        return report
    
    def save_reports(self, report: str, processed_articles: List[Dict]):
        """Salva relatórios em TXT e JSON"""
        # Salvar TXT
        try:
            with open("noticias_resumidas.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            print("✓ Relatório salvo em: noticias_resumidas.txt")
        except Exception as e:
            print(f"✗ Erro ao salvar TXT: {e}")
            
        # Salvar JSON
        try:
            with open("noticias_resumidas.json", 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, ensure_ascii=False, indent=2)
            print("✓ Dados JSON salvos em: noticias_resumidas.json")
        except Exception as e:
            print(f"✗ Erro ao salvar JSON: {e}")

    def run(self, use_newsapi: bool = True, use_g1: bool = True):
        """Executa o pipeline completo"""
        print("\n🔍 Iniciando busca de notícias...\n")
        all_articles = []
        
        if use_newsapi:
            all_articles.extend(self.fetch_news_newsapi("Brasil", days=1))
        
        if use_g1:
            all_articles.extend(self.fetch_news_g1())
        
        if not all_articles:
            print("✗ Nenhuma notícia encontrada nas fontes selecionadas.")
            return
        
        print(f"\n📊 Processando e sumarizando {len(all_articles)} notícias...\n")
        processed_articles = self.process_articles(all_articles)
        
        print("\n✓ Processamento concluído!\n")
        report = self.generate_report(processed_articles)
        print(report)
        
        self.save_reports(report, processed_articles)


def main():
    print("=" * 60)
    print("   AGREGADOR E SUMARIZADOR DE NOTÍCIAS DO BRASIL")
    print("=" * 60)
    
    aggregator = NewsAggregator()
    aggregator.run(use_newsapi=True, use_g1=True)

if __name__ == "__main__":
    main()
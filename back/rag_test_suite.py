"""
IceMan RAG — Bateria de Testes de Acurácia
Categorias: Preços | Estoque | Comparativas | Disponibilidade | Políticas | Negativas
Pontuação por resposta: 2 = correto | 1 = parcial | 0 = errado
"""

import requests
import json
import time
import re

API_URL = "http://localhost:8000/chat"

# ─────────────────────────────────────────────────────────────────────────────
# SUITE DE TESTES  (pergunta, resposta_esperada, palavras_chave_obrigatórias)
# Palavras-chave: TODAS precisam estar presentes na resposta para pontuar 2.
# Se ao menos metade estiver, pontua 1. Caso contrário, 0.
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [

    # ── CATEGORIA 1: PREÇOS ESPECÍFICOS ───────────────────────────────────────
    {
        "categoria": "Preço Específico",
        "pergunta": "Qual é o preço do Rolex Submariner Date?",
        "esperado": "BRL 96.000",
        "keywords": ["96.000", "96000"],
    },
    {
        "categoria": "Preço Específico",
        "pergunta": "Quanto custa o Omega Speedmaster Moonwatch?",
        "esperado": "BRL 38.500",
        "keywords": ["38.500", "38500"],
    },
    {
        "categoria": "Preço Específico",
        "pergunta": "Qual é o preço do Audemars Piguet Royal Oak 37mm Ouro Rosa?",
        "esperado": "BRL 285.000",
        "keywords": ["285.000", "285000"],
    },
    {
        "categoria": "Preço Específico",
        "pergunta": "Quanto custa o TAG Heuer Formula 1 Chronograph?",
        "esperado": "BRL 12.900",
        "keywords": ["12.900", "12900"],
    },
    {
        "categoria": "Preço Específico",
        "pergunta": "Qual é o preço do Panerai Luminor Marina Titanium?",
        "esperado": "BRL 62.000",
        "keywords": ["62.000", "62000"],
    },
    {
        "categoria": "Preço Específico",
        "pergunta": "Qual é o preço do Rolex Datejust 41 Mostrador Azul?",
        "esperado": "BRL 75.400",
        "keywords": ["75.400", "75400"],
    },

    # ── CATEGORIA 2: ESTOQUE ESPECÍFICO ───────────────────────────────────────
    {
        "categoria": "Estoque Específico",
        "pergunta": "Quantas unidades do Rolex Cosmograph Daytona estão disponíveis?",
        "esperado": "4 unidades",
        "keywords": ["4"],
    },
    {
        "categoria": "Estoque Específico",
        "pergunta": "Qual é o estoque do Omega Seamaster Diver 300M Azul?",
        "esperado": "24 unidades",
        "keywords": ["24"],
    },
    {
        "categoria": "Estoque Específico",
        "pergunta": "Quantas unidades do TAG Heuer Aquaracer Professional 300 estão em estoque?",
        "esperado": "42 unidades",
        "keywords": ["42"],
    },
    {
        "categoria": "Estoque Específico",
        "pergunta": "Quantas unidades do Rolex GMT-Master II Batgirl estão disponíveis?",
        "esperado": "8 unidades",
        "keywords": ["8"],
    },
    {
        "categoria": "Estoque Específico",
        "pergunta": "Qual é a quantidade em estoque do Panerai Luminor Submersible 42mm?",
        "esperado": "19 unidades",
        "keywords": ["19"],
    },

    # ── CATEGORIA 3: COMPARATIVAS / SUPERLATIVAS ──────────────────────────────
    {
        "categoria": "Comparativa",
        "pergunta": "Qual é o produto mais caro do catálogo?",
        "esperado": "Rolex Daytona Platinum Exclusivo — BRL 697.000",
        "keywords": ["697.000", "Platinum", "Daytona"],
    },
    {
        "categoria": "Comparativa",
        "pergunta": "Qual é o produto mais barato disponível na IceMan?",
        "esperado": "TAG Heuer Formula 1 Chronograph — BRL 12.900",
        "keywords": ["12.900", "Formula 1"],
    },
    {
        "categoria": "Comparativa",
        "pergunta": "Qual produto possui a maior quantidade em estoque?",
        "esperado": "TAG Heuer Formula 1 Chronograph — 49 unidades",
        "keywords": ["49", "Formula 1"],
    },
    {
        "categoria": "Comparativa",
        "pergunta": "Qual produto tem o menor estoque (excluindo encomendas)?",
        "esperado": "2 unidades — AP Royal Oak Ouro Rosa, TAG Monaco Flyback ou Panerai Carbotech",
        "keywords": ["2"],
    },
    {
        "categoria": "Comparativa",
        "pergunta": "Qual é o Rolex mais caro do catálogo?",
        "esperado": "Rolex Daytona Platinum Exclusivo — BRL 697.000",
        "keywords": ["697.000", "Platinum"],
    },
    {
        "categoria": "Comparativa",
        "pergunta": "Qual é o relógio mais barato da Omega?",
        "esperado": "Omega Seamaster Diver 300M Azul — BRL 34.900",
        "keywords": ["34.900", "Seamaster"],
    },

    # ── CATEGORIA 4: DISPONIBILIDADE POR LOJA ─────────────────────────────────
    {
        "categoria": "Disponibilidade",
        "pergunta": "Em quais lojas posso encontrar o Rolex Datejust 41?",
        "esperado": "Ibirapuera, Paulista, Leblon, Barra e Campinas",
        "keywords": ["Ibirapuera", "Paulista", "Leblon", "Barra", "Campinas"],
    },
    {
        "categoria": "Disponibilidade",
        "pergunta": "O Rolex Cosmograph Daytona está disponível na IceMan Ibirapuera?",
        "esperado": "Não — apenas Paulista e Barra",
        "keywords": ["Paulista", "Barra"],
    },
    {
        "categoria": "Disponibilidade",
        "pergunta": "Onde posso comprar o Panerai Luminor Marina Titanium?",
        "esperado": "IceMan Ibirapuera, Leblon e Campinas",
        "keywords": ["Ibirapuera", "Leblon", "Campinas"],
    },
    {
        "categoria": "Disponibilidade",
        "pergunta": "Qual é o telefone da loja IceMan Campinas?",
        "esperado": "(19) 3456-9000",
        "keywords": ["3456-9000", "19"],
    },
    {
        "categoria": "Disponibilidade",
        "pergunta": "Quem é a gerente da loja IceMan Ibirapuera?",
        "esperado": "Carla Mendes",
        "keywords": ["Carla", "Mendes"],
    },

    # ── CATEGORIA 5: POLÍTICAS ─────────────────────────────────────────────────
    {
        "categoria": "Política",
        "pergunta": "Qual é o prazo de garantia dos relógios vendidos pela IceMan?",
        "esperado": "2 anos de garantia do fabricante",
        "keywords": ["2 anos"],
    },
    {
        "categoria": "Política",
        "pergunta": "Existe desconto para pagamento via Pix?",
        "esperado": "Sim, 5% de desconto",
        "keywords": ["5", "desconto"],
    },
    {
        "categoria": "Política",
        "pergunta": "Qual é o prazo para devolução de um relógio?",
        "esperado": "30 dias para relógios não utilizados na embalagem original",
        "keywords": ["30 dias"],
    },
    {
        "categoria": "Política",
        "pergunta": "O serviço de gravação é gratuito?",
        "esperado": "Gratuito para compras acima de BRL 50.000, caso contrário custa BRL 150",
        "keywords": ["150", "50.000"],
    },
    {
        "categoria": "Política",
        "pergunta": "Em quantas parcelas posso pagar sem juros no cartão de crédito?",
        "esperado": "Até 6 parcelas sem juros",
        "keywords": ["6"],
    },

    # ── CATEGORIA 6: NEGATIVAS / FORA DE ESCOPO ───────────────────────────────
    {
        "categoria": "Negativa",
        "pergunta": "A IceMan vende relógios Cartier?",
        "esperado": "Não — apenas Rolex, Audemars Piguet, Omega, TAG Heuer e Panerai",
        "keywords": ["não", "Cartier"],
    },
    {
        "categoria": "Negativa",
        "pergunta": "Vocês vendem anéis ou colares?",
        "esperado": "Não — a IceMan vende exclusivamente relógios",
        "keywords": ["não"],
    },
    {
        "categoria": "Negativa",
        "pergunta": "O Rolex Daytona Platinum está disponível imediatamente para retirada?",
        "esperado": "Não — está sob encomenda com entrega em 90 dias",
        "keywords": ["90 dias", "encomenda"],
    },
    {
        "categoria": "Negativa",
        "pergunta": "Vocês têm Audemars Piguet Royal Oak Offshore Diver disponível na IceMan Campinas?",
        "esperado": "Não — disponível apenas em Paulista e Barra",
        "keywords": ["não", "Paulista", "Barra"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text):
    return text.lower().replace(".", "").replace(",", "").replace("-", "").replace("_", "")

def score_answer(answer: str, keywords: list) -> int:
    a = normalize(answer)
    hits = sum(1 for kw in keywords if normalize(kw) in a)
    if hits == len(keywords):
        return 2
    elif hits >= max(1, len(keywords) // 2):
        return 1
    return 0

def run_tests(cases):
    results = []
    categories = {}

    print("=" * 80)
    print("  ICEMAN RAG — BATERIA DE TESTES DE ACURÁCIA")
    print("=" * 80)

    for i, tc in enumerate(cases, 1):
        cat = tc["categoria"]
        q   = tc["pergunta"]
        exp = tc["esperado"]
        kws = tc["keywords"]

        try:
            resp = requests.post(API_URL, json={"question": q}, timeout=40)
            data = resp.json()
            answer = data.get("answer", "[SEM RESPOSTA]")
            sources_count = len(data.get("sources", []))
        except Exception as e:
            answer = f"[ERRO: {e}]"
            sources_count = 0

        pts = score_answer(answer, kws)
        label = "✅ CORRETO " if pts == 2 else ("⚠️  PARCIAL " if pts == 1 else "❌ ERRADO  ")

        # Acumular por categoria
        if cat not in categories:
            categories[cat] = {"total": 0, "score": 0}
        categories[cat]["total"] += 2
        categories[cat]["score"] += pts

        results.append({
            "id": i,
            "categoria": cat,
            "pergunta": q,
            "esperado": exp,
            "resposta": answer,
            "pts": pts,
            "fontes": sources_count
        })

        print(f"\n[{i:02d}] {label} | {cat}")
        print(f"  P: {q}")
        print(f"  E: {exp}")
        print(f"  R: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"  Pontos: {pts}/2  |  Fontes recuperadas: {sources_count}")

        time.sleep(0.8)   # pequena pausa entre requisições

    return results, categories


def print_report(results, categories):
    total_score = sum(r["pts"] for r in results)
    max_score   = len(results) * 2
    accuracy    = (total_score / max_score) * 100

    print("\n")
    print("=" * 80)
    print("  RELATÓRIO FINAL DE ACURÁCIA")
    print("=" * 80)

    # Por categoria
    print("\n  PONTUAÇÃO POR CATEGORIA")
    print("  " + "-" * 50)
    for cat, data in categories.items():
        pct = (data["score"] / data["total"]) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:<22} [{bar}] {pct:5.1f}%  ({data['score']}/{data['total']})")

    # Resumo geral
    print(f"\n  {'─'*50}")
    corretos  = sum(1 for r in results if r["pts"] == 2)
    parciais  = sum(1 for r in results if r["pts"] == 1)
    errados   = sum(1 for r in results if r["pts"] == 0)
    print(f"  ✅ Corretos  : {corretos:>3}  ({corretos/len(results)*100:.1f}%)")
    print(f"  ⚠️  Parciais  : {parciais:>3}  ({parciais/len(results)*100:.1f}%)")
    print(f"  ❌ Errados   : {errados:>3}  ({errados/len(results)*100:.1f}%)")
    print(f"  {'─'*50}")
    print(f"  ACURÁCIA GERAL: {total_score}/{max_score} pontos → {accuracy:.1f}%")
    print("=" * 80)

    # Detalhamento dos erros
    if errados > 0 or parciais > 0:
        print("\n  RESPOSTAS INCORRETAS / PARCIAIS")
        print("  " + "-" * 50)
        for r in results:
            if r["pts"] < 2:
                tag = "⚠️  PARCIAL" if r["pts"] == 1 else "❌ ERRADO "
                print(f"\n  [{r['id']:02d}] {tag}  [{r['categoria']}]")
                print(f"  P: {r['pergunta']}")
                print(f"  E: {r['esperado']}")
                print(f"  R: {r['resposta'][:300]}")

    print()


if __name__ == "__main__":
    results, categories = run_tests(TEST_CASES)
    print_report(results, categories)

# Consumo Energético - Forecasting com SARIMAX e Chronos

Projeto de previsão de consumo energético brasileiro utilizando:

- SARIMAX (modelo estatístico clássico)
- Chronos (Foundation Model da Amazon)

O objetivo é comparar o desempenho dos modelos em séries temporais energéticas sob diferentes disponibilidades históricas de dados.

---

# Dataset

O projeto utiliza o dataset:

```text
Dados_abertos_Consumo_Mensal.csv
```

Estrutura esperada:

```csv
Data;Regiao;Sistema;Classe;TipoConsumidor;Consumo
01/03/2026;Centro-Oeste;SUDESTE / CENTRO - OESTE;Residencial;Cativo;1508165,7
```

---

# Requisitos

## Python

Recomendado utilização de Python Python 3.11

---

# Criar ambiente virtual

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

# Instalação das dependências

## SARIMAX

```bash
pip install -r requirements.txt
```

---

# Estrutura do projeto

```text
.
├── Dados_abertos_Consumo_Mensal.csv
├── previsao_consumo_sarimax.py
├── previsao_consumo_chronos.py
├── requiriments.txt
└── README.md
```

---

# Configuração dos filtros

Os filtros podem ser alterados diretamente no código:

```python
classe = "Residencial"
tipo_consumidor = "Cativo"
```

---

# Configuração de Periodo

## Período total utilizado

```python
data_inicial = "2010-01-01"
data_final = "2026-03-01"
```

Define o intervalo completo da série temporal (Periodo processado).

---

## Data limite de treino

```python
data_limite_treino = "2023-11-14"
```

Define:
- até onde o modelo aprende.
- Dia posterior, igual início do período de dados de teste.

---

# Executando o modelo SARIMAX

```bash
python previsao_consumo_sarimax.py
```

Modelo estatístico interpretável e eficiente em séries estruturadas e sazonais. Executar comando acima em terminal.
---

# Executando o modelo Chronos

```bash
python previsao_consumo_chronos.py
```

Foundation Model baseado em deep learning temporal pré-treinado em múltiplas séries temporais. Executar comando acima em terminal.
---

# Métricas utilizadas

## MAE
Erro absoluto médio.

---

## RMSE
Erro quadrático médio.

---

## MAPE
Erro percentual médio.

Principal métrica utilizada na comparação dos modelos.

---

# Licença

Projeto acadêmico desenvolvido para fins de pesquisa e estudo em previsão de séries temporais energéticas.
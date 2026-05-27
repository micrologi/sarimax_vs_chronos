import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from chronos import ChronosPipeline

arquivo_csv = "Dados_abertos_Consumo_Mensal.csv"

# filtros
classe = "Residencial"
tipo_consumidor = "Cativo"

#periodo de estabilidade:
estabilidade_quebra = input('1-Estabilidade / 2-Quebra estrutural (Pandemia): ')

data_inicial = "2004-01-01"
if estabilidade_quebra == "1":
    data_limite_treino = "2017-12-31"
    data_final = "2019-12-31"
else: #quebra estrutural
    data_limite_treino = "2019-12-31"
    data_final = "2021-12-31"


print("Carregando dados...")

df = pd.read_csv(
    arquivo_csv,
    sep=";",
    encoding="utf-8"
)

df["Data"] = pd.to_datetime(
    df["Data"],
    format="%d/%m/%Y",
    errors="coerce"
)

df["Data"] = (
    df["Data"].dt.to_period("M").dt.to_timestamp()
)

df["Consumo"] = (
    df["Consumo"].astype(str).str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["Consumo"] = pd.to_numeric(
    df["Consumo"],
    errors="coerce"
)

df = df.dropna(
    subset=["Data", "Consumo"]
)

dt_ini = (
    pd.to_datetime(data_inicial).to_period("M").to_timestamp()
)

dt_fim = (
    pd.to_datetime(data_final).to_period("M").to_timestamp()
)

df = df[
    (df["Data"] >= dt_ini) & (df["Data"] <= dt_fim)
].copy()

#df = df[
#    (df["Classe"] == classe) & (df["TipoConsumidor"] == tipo_consumidor)
#].copy() 

serie = df[
    ["Data", "Consumo"]
].copy()

serie.columns = ["ds", "y"]

serie = (
    serie.groupby("ds")["y"].sum().reset_index()
)

serie = serie.sort_values("ds")
serie = serie.dropna()
serie = serie[
    serie["y"] > 0
]

data_limite = (
    pd.to_datetime(data_limite_treino).to_period("M").to_timestamp()
)

treino = serie[
    serie["ds"] <= data_limite
].copy()

teste = serie[
    serie["ds"] > data_limite
].copy()

print("Treino:")
print(treino.tail())

print("Teste:")
print(teste.head())

y_train = treino["y"].values.astype(np.float32)
horizon = len(teste)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {device}")

print("Carregando modelo Chronos...")

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device,
    torch_dtype=torch.float32
)

print("Modelo carregado.")

context = torch.tensor(y_train)

print("Executando previsão...")

forecast = pipeline.predict(
    context,
    prediction_length=horizon,
    num_samples=20
)

forecast_mean = forecast.mean(dim=1).numpy().flatten()

resultado = pd.DataFrame({
    "Data": teste["ds"].values,
    "Real": teste["y"].values,
    "Previsto": forecast_mean[:len(teste)]
})

print("\nReal vs previsto:")
print(resultado)

mae = mean_absolute_error(
    resultado["Real"],
    resultado["Previsto"]
)

rmse = np.sqrt(
    mean_squared_error(
        resultado["Real"],
        resultado["Previsto"]
    )
)

mape = np.mean(
    np.abs(
        (resultado["Real"] - resultado["Previsto"]) / resultado["Real"]
    )
) * 100

print("\nMétricas:")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"MAPE : {mape:.2f}%")

# continuidade visual
datas_real = pd.concat([
    treino["ds"].tail(1),
    resultado["Data"]
])

valores_real = pd.concat([
    treino["y"].tail(1),
    resultado["Real"]
])

datas_prev = pd.concat([
    treino["ds"].tail(1),
    resultado["Data"]
])

valores_prev = pd.concat([
    treino["y"].tail(1),
    resultado["Previsto"]
])

plt.figure(figsize=(16, 7))

plt.plot(
    treino["ds"],
    treino["y"],
    label="Treino",
    color="blue",
    linewidth=2
)

plt.plot(
    datas_real,
    valores_real,
    label="Real",
    color="green",
    linewidth=2
)

plt.plot(
    datas_prev,
    valores_prev,
    label="Previsto",
    color="red",
    linestyle="--",
    linewidth=2
)

plt.axvline(
    x=data_limite,
    color="black",
    linestyle=":",
    linewidth=2,
    label="Início teste"
)

plt.title(
    f"""
    Chronos - Consumo energético
    """
)

plt.xlabel("Data")
plt.ylabel("Consumo")
plt.grid(True)
plt.legend()
plt.show()
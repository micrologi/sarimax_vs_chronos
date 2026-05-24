import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

arquivo_csv = "Dados_abertos_Consumo_Mensal.csv"

# filtros
classe = "Residencial"
tipo_consumidor = "Cativo"

# período total utilizado
data_inicial = "2004-01-01"
data_final = "2025-12-31"
data_limite_treino = "2024-12-31"

# parâmetros do modelo
ordem = (2, 1, 2)
ordem_sazonal = (1, 1, 1, 12)

print("Carregando dados...")

df = pd.read_csv(
    arquivo_csv,
    sep=";",
    encoding="utf-8"
)

# formtação das colunas
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

# filtros
df = df[
    (df["Data"] >= dt_ini) & (df["Data"] <= dt_fim)
].copy()

df = df[
    (df["Classe"] == classe) & (df["TipoConsumidor"] == tipo_consumidor)
].copy()

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

print("\nTeste:")
print(teste.head())

treino = treino.set_index("ds")
teste = teste.set_index("ds")

print("Treinando modelo...")

modelo = SARIMAX(
    treino["y"],
    order=ordem,
    seasonal_order=ordem_sazonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

modelo_fit = modelo.fit(disp=False)

print("Modelo treinado.")

forecast = modelo_fit.get_forecast(
    steps=len(teste)
)

forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

resultado = pd.DataFrame({

    "Data": teste.index,
    "Real": teste["y"].values,
    "Previsto": forecast_mean.values,
    "Lim.Inferior": forecast_ci.iloc[:, 0].values,
    "Lim.Superior": forecast_ci.iloc[:, 1].values

})

print("Real vs previsto:")
print(resultado)

# métricas de erros
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

print("Métricas:")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"MAPE : {mape:.2f}%")

# continuidade visual
datas_real = pd.concat([
    treino.reset_index()["ds"].tail(1),
    resultado["Data"]
])

valores_real = pd.concat([
    treino["y"].tail(1),
    resultado["Real"]
])

datas_prev = pd.concat([
    treino.reset_index()["ds"].tail(1),
    resultado["Data"]
])

valores_prev = pd.concat([
    treino["y"].tail(1),
    resultado["Previsto"]
])

plt.figure(figsize=(16, 7))

plt.plot(
    treino.index,
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

plt.fill_between(
    resultado["Data"],
    resultado["Lim.Inferior"],
    resultado["Lim.Superior"],
    color="red",
    alpha=0.15
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
    SARIMAX - Consumo energético
    Classe: {classe}
    Tipo: {tipo_consumidor}
    """
)

plt.xlabel("Data")
plt.ylabel("Consumo")
plt.grid(True)
plt.legend()
plt.show()

print("\nResumo estatístico:\n")
print(modelo_fit.summary())
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv")  

cols_remover = ["Name", "RescuerID", "Description", "PetID", "AdoptionSpeed"]
df = df.drop(columns=cols_remover)

df = df.dropna() # removendo linhas com valores ausentes

scaler = StandardScaler() # normalização que a cris gosta
df_normalizado = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

df_normalizado.to_csv("train_tratado.csv", index=False)

print("Arquivo tratado salvo como train_tratado.csv")

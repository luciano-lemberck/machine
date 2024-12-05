#Projeto COVID-19
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px 
import plotly.graph_objects as go
import csv
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from prophet import Prophet as Prophet


# importar os dados do github
url = 'https://github.com/luciano-lemberck/machine/blob/54016445e2c95de9a5ac8937143d111029c838cf/covid_19_data_v3.csv?raw=true'

dados = pd.read_csv(url, sep=",")
#, parse_dates=['observationdate', 'lastupdate'])
#dados = pd.read_csv(url, sep=",")
print(dados)

#dados.dtypes


#contando a quantidade por pais
#dados = dados.countryregion.value_counts()
#print(dados)

#listando somente o brasil
#brasil_filtrado = dados.loc[dados['countryregion'] == 'Brazil']

#casos confirmados
brasil_confirmado = dados.loc[(dados.countryregion == 'Brazil') & (dados.confirmed > 0)]
print(brasil_confirmado)

#gráfico da evolução de casos confirmados
grafico_br_conf = px.line(brasil_confirmado, x='observationdate', y='confirmed', title='Casos confirmados no Brasil')
grafico_br_conf.show()


#Novos casos por dia
brasil_confirmado['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil_confirmado['confirmed'].iloc[x] - brasil_confirmado['confirmed'].iloc[x-1], 
    np.arange(brasil_confirmado.shape[0])
))

#gráfico de novos casos por dia
grafico_novos_casos= px.line(brasil_confirmado, x='observationdate', y='novoscasos', title='Novos casos por dia')
print(brasil_confirmado)
grafico_novos_casos.show()

#gráfico mortes covid-19 brasil
#séries temporais
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=brasil_confirmado.observationdate, y=brasil_confirmado.deaths, name='Mortes',
               mode='lines+markers', line={'color': 'red'})
)

#layout
fig.update_layout(title='Mortes por COVID-19 no Brasil')
fig.show()

'''
#Taxa de crescimento
#regra matemática -> taxa_crescimento = (presente/passado)**(1/n)-1
def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
#se data inicio for none, define como a primeira data disponível
      
    if data_inicio == None:
        data_inicio = pd.to_datetime(data.observationdate.loc[data[variable] > 0].min())
    else:
        data_inicio = pd.to_datetime(data_inicio)
    if data_fim == None:
        data_fim = pd.to_datetime(data.observationdate.iloc[-1])
    else:
        data_fim = pd.to_datetime(data_fim)
        
    #Define os valores do presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0] #teve erro aqui e nao consegui corrigir
    
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    #Define o número de pontos no tempo que vamos avaliar
    n = abs(data_fim - data_inicio).days 
    
        #Calcular a taxa
    taxa = (presente/passado)**(1/n) - 1
    return taxa*100

#Taxa de crescinmento médio do COVID no Brasil em todo o período
taxa_crescimento(brasil_confirmado, 'confirmed') 
print(taxa_crescimento)
'''

'''
#Taxa de crescimento diária
def taxa_crescimento_diaria(brasil_confirmado, variable, data_inicio=None):
    #se data inicio for none, define como a primeira data disponível
    if data_inicio == None:
        data_inicio = pd.to_datetime(brasil_confirmado.observationdate.loc[brasil_confirmado[variable] > 0].min())
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    data_fim = brasil_confirmado.observationdate.max()
    #Define o número de pontos no tempo que vamos avaliar
    n = abs(data_fim - data_inicio).days 
    
    # taxa calculada de um dia para outro
    taxas = list(map(
        lambda x: (brasil_confirmado[variable].iloc[x] - brasil_confirmado[variable].iloc[x-1]) / brasil_confirmado[variable].iloc[x-1], 
        range(1, n+1)
    ))
    return np.array(taxas) * 100

tx_dia = taxa_crescimento_diaria(brasil_confirmado, 'confirmed')
print(tx_dia)
'''


# Predições -> machine learning

confirmados = brasil_confirmado.confirmed
brasil_confirmado['observationdate'] = pd.to_datetime(brasil_confirmado['observationdate'], dayfirst=True)
brasil_confirmado.set_index('observationdate', inplace=True)
brasil_confirmado = brasil_confirmado.asfreq('D')

#decompondo a serie temporal para entender o que ela tem de tendência, sazonalidade e ruido	
res = seasonal_decompose(confirmados, period=30, model= 'additive') 
res.plot()

# Plotar os componentes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
res.observed.plot(ax=ax1) #plota os dados observados no primeiro subgráfico (ax1).
ax1.set_title('Observado')
res.trend.plot(ax=ax2) #plota a tendência no segundo subgráfico (ax2).
ax2.set_title('Tendência')
res.seasonal.plot(ax=ax3) #plota a sazonalidade no terceiro subgráfico (ax3).
ax3.set_title('Sazonalidade')
res.resid.plot(ax=ax4) #plota o resíduo no quarto subgráfico (ax4).
ax4.set_title('Resíduo')
plt.tight_layout() #ajustar o layout para não sobrepor os gráficos
plt.show()



#MODELAGEM - vamos olhar para o passado e estimar o futuro
modelo = auto_arima(confirmados)

fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observados'
))
fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
))
fig.add_trace(go.Scatter(
    x=pd.date_range('01/04/2020', '01/05/2020'), y=modelo.predict(31), name='Forecast'
))
fig.update_layout(title='Predição de casos confirmados no Brasil para os proximos 30 dias')
fig.show()
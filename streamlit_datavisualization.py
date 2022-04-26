#IMPORTAÇÃO DAS BIBLIOTECAS===================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from st_aggrid import AgGrid 
import plotly.express as px
import datetime as dt
#Bibliotecas de classes de machine learning 
from sklearn.cluster import KMeans #Importa o Kmenas 
from sklearn.model_selection import train_test_split  #Divide as instancias entre treino e teste
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Transforma variaveis categoricas em numericas
from sklearn.datasets import make_classification  #ANYMORE
from sklearn.inspection import permutation_importance # Importa o algoritmo de permutação de importancia 
from sklearn.preprocessing import StandardScaler #Importa o algitmo "padronizador"
from sklearn.utils import resample #Reamostragem
#Variaveis correspondetes aos devidos endereços dos dataset selecionados======================================================================
#O conjunto de dados foi cópiado para o GitHub de um dos integrantes da equipe Cosmus, responsavel pelo desenvolvimento do artigo
products_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_products_dataset.csv') 
orders_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_orders_dataset.csv')
category_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/product_category_name_translation.csv')
customer_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_customers_dataset.csv')
payment_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_order_payments_dataset.csv')
review_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_order_reviews_dataset.csv')
items_ds = pd.read_csv('https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_order_items_dataset.csv')
#PRÉ-PROCESSAMENTO DE DADOS===================================================================================================================
#Algoritmo para obter as contagens de vezes que cada produtos foi vendido --------------------------------------------------------------------
sold_count = items_ds['product_id'].value_counts() #Cria uma variavel em arquivo "Serie" que contem o número de vendas dos produtos
sold_count = sold_count.to_frame() #Converte o arquivo Serie para Dataframe
sold_count.reset_index(inplace=True) #Cria um index para esse pequeno Dataframe
sold_count.rename(columns={'index':'product_id','product_id':'sold_count'}, inplace=True) #Renomeia as colunas
#Juntando e sincrozizando os novos atributos ao dataset "product_ds"
products_ds = pd.merge(products_ds, sold_count) #Junta e sincroniza os atributos do dataset "sold_count" ao "products_ds"
#Algoritmo para obter as contagens de vezes que os clientes compraram  -----------------------------------------------------------------------
frequency = customer_ds['customer_unique_id'].value_counts() #Cria uma variavel em arquivo "Serie" que contem o número das vezes que os clientes compraram
frequency = frequency.to_frame()
frequency.reset_index(inplace=True)
frequency.rename(columns={'index':'customer_unique_id','customer_unique_id':'frequency'}, inplace=True)
#Juntando e sincrozizando os novos atributos ao dataset "customer_ds"
customer_ds = pd.merge(customer_ds, frequency)
#PERGUNTA 1 --------------------------------------------------------------------------------------------------------------------------------
#Copiando o dataset "products_ds" para usar na pergunta 1.
prod_p1 = products_ds.copy()
#Removendo os atributos que não são relevantes para a analise
prod_p1.drop(['product_weight_g','product_length_cm','product_height_cm','product_width_cm'], axis=1, inplace=True)
#Verificando os valores NaN do dataset. 
check_nan = prod_p1[prod_p1.isna().any(axis=1)] #Seleciona apenas as colunas com valores NaN
print(prod_p1.isnull().sum()) #Verifica quantas linhas possui valores NaN
#Removendo as linhas NaN
prod_p1 = prod_p1.dropna() #Remove as linhas que possuem valores NaN
print(prod_p1.isnull().sum()) #Verifica quantas linhas possui valores NaN
#Organizando o dataset de forma decrescente
prod_p1 = prod_p1.sort_values(by=['sold_count'], ascending=False, ignore_index=True) # Organiza a lista de forma decrescente, usando a coluna 'sold' em consideração
#PERGUNTA 2 ----------------------------------------------------------------------------------------------------------------------------------
#Selecionando as colunas com atributos de interesse
cust_p2 = customer_ds[['customer_id','customer_unique_id','frequency']] #Seleciona os atributos do "customer_ds"
orde_p2 = orders_ds[['order_id','order_purchase_timestamp','customer_id']] #Seleciona os atributos do "orders_ds"
paym_p2 = payment_ds[['order_id','payment_value']] #Seleciona os atributos do "payments_ds"
#Unindo os dataset
p2_ds = pd.merge(cust_p2,orde_p2) #Unindo o dataset "cust_p2" ao "orde_p2"
p2_ds = pd.merge(p2_ds,paym_p2) #Unindo o dataset "paym_p2" ao "p2_ds"
#Criando variavel que contêm o timestamp atual
current_timestamp = dt.datetime.now()
#current_timestamp= current_timestamp/np.timedelta64(1,'h')
#Criando coluna que diz o quão recente foi a compra daquele customer
p2_ds['recent_pur'] = current_timestamp - p2_ds['order_purchase_timestamp'] #Calcula a diferença entre o timestam atual e a data da compra
p2_ds['recent_pur'] = p2_ds['recent_pur']/np.timedelta64(1,'h') #Transforma o timestamp em somente horas
#Exclui as colunas 'customer_id', 'order_id' e 'order_purchase_timestamp'
p2_ds.drop(['customer_id', 'customer_unique_id', 'order_id', 'order_purchase_timestamp'], axis=1, inplace=True)
#Padronizando a média de todos os dados para 0 
scaler = StandardScaler() #Instancia a função responsavel
p2_ds = scaler.fit_transform(p2_ds) #Usa a função e padroniza
#Transformação do "p2_ds" para o grafico
p2_ds = pd.DataFrame(p2_ds) #Transporma o "p2_ds" novamente em Dataframe
p2_ds = pd.DataFrame.rename(p2_ds, columns={0:'frequency', 1:'payment_value',	2:'recency'}) #retorna o nome das colunas
#PERGUNTA 3 --------------------------------------------------------------------------------------------------------------------------------
#Copiando o dataset "orders_ds" e apagando instancias NaN
p3_ds = orders_ds.copy() #Copia o dataset "orders_ds"
p3_ds = p3_ds.dropna() #Remove as linhas que possuem valores NaN
#corrige o valor de frete para o pedido + cria coluna com quantidade de itens + corrige valor total do pedido somando o valor de todos os produtos
items_p3 = items_ds.drop(['product_id','seller_id'],axis=1)
# cria um dicionários com as funções que serão executadas para cada coluna, combinando as linhas que pertencem ao mesmo id do pedido
functions = {'order_item_id':'count', # transforma na quantidade de itens no pedido
             'shipping_limit_date':'first', # permanece igual
             'price':'sum', # soma o valor dos produtos do pedido
             'freight_value':'sum'} # valor total do frete no pedido
items_p3 = items_p3.groupby(['order_id'], as_index=False).aggregate(functions)
items_p3.rename(columns = {'order_item_id':'quantity_of_items'}, inplace = True)
items_p3['shipping_limit_date'] = pd.to_datetime(items_ds['shipping_limit_date'], format='%Y-%m-%d %H:%M:%S') #transformando em timestamp
#Unindo os dataset
p3_ds = pd.merge(p3_ds, payment_ds) #Unindo o dataset "payments_ds" ao "p3_ds"
p3_ds = pd.merge(p3_ds, review_ds) #Unindo o dataset "review_ds" ao "p3_ds"
p3_ds = pd.merge(p3_ds, items_p3) #Unindo o dataset "items_ds" ao "p3_ds"
#Removendo instancias e atributos irrelevantes
#p3_ds = p3_ds.drop_duplicates(subset='order_id') #Remove as instancias duplicadas vindas do dataset ITEMS.
p3_ds = p3_ds.drop(['order_id','review_id','customer_id','review_comment_title','review_comment_message'],axis=1) 
#Remove as colunas irrelevantes
#MINERAÇÃO DE DADOS --------------------------------------------------------------------------------------------------------------------------
def grafico_elbow():
  elbow_df = pd.DataFrame({'Clusters': K, 'within-clusters sum-of-squares': distortions})
  fig = (px.line(elbow_df, x='Clusters', y='within-clusters sum-of-squares', template='seaborn')).update_traces(mode='lines+markers')
  st.plotly_chart(fig)

# STREAMLIT VISUALIZATION=====================================================================================================================
st.title("Cosmus - Visualização de Dados")
#cria barra lateral
st.sidebar.title("Menu")
#cria uma selectbox para navegar entre as páginas de análises
select_page = st.sidebar.selectbox("Selecionar sessão",['Análise exploratória de dados','Mineração de dados',])
#função que carrega o template de cada página de análise, recebe a lista de datasets e o indice correspondente
def load_page(data):
    st.subheader(select_page) #subtítulo
    df = pd.read_csv(data) #leitura do dataset
    AgGrid(df)  #cria dataframe interativo(filtros etc.)
    return df
#os if statements abaixo referem-se as opções do selectbox
if select_page == 'Order Payments':
    df = load_page(order_payment) # chama funçao de template
    pt_count = df['payment_type'].value_counts() #retorna frequencia de valores
    fig = px.bar(pt_count) #plotagem do gráfico
    st.plotly_chart(fig, use_container_width=True)
#repete a mesma estrutura da condição anterior
elif select_page =='Sellers list':
    df = load_page(sellers)
    pt_count = df['seller_state'].value_counts()
    fig = px.bar(pt_count)
    st.plotly_chart(fig)
# cria mais um selectbox para selecionar uma categoria de produto, e plota a distribuição de pesos(g) dos produtos num scatterplot
elif select_page =='Products list':
    df = load_page(product)
    category_filter = st.selectbox('Select product category',['cds_dvs_musicais','casa_conforto_2','moveis_quarto','fashion_roupa_infanto_juvenil', 'livros_importados', 'alimentos_bebidas', 'construcao_ferramentas_jardim', 'artigos_de_festa', 'telefonia_fixa', 'fashion_underwear_e_moda_praia', 'climatizacao', 'eletroportateis', 'casa_construcao', 'telefonia', 'moveis_escritorio', 'informatica_acessorios', 'eletrodomesticos', 'utilidades_domesticas', 'perfumaria' ])
    category_selected = df[df['product_category_name']==category_filter]
    fig = px.scatter(category_selected, y="product_category_name", x="product_weight_g")
    st.plotly_chart(fig, use_container_width=True)
#mostra a frequencia de cada categoria de produto
    st.subheader('Products by category')
    category_count = df['product_category_name'].value_counts()
    fig1 = px.bar(category_count, width=800, height=500)
    st.plotly_chart(fig1)
    description = df.describe() #descreve resumidamente os dados
    st.dataframe(description)
#mostra frequência de avaliações (score 0 -5)
elif select_page =='Order reviews':
    df = load_page(order_review)
    score_count = df['review_score'].value_counts()
    fig2 = px.bar(score_count, width=800, height=500)
    st.plotly_chart(fig2)
#mostra os preços dos produtos e seus respectivos fretes
elif select_page =='Order item':
    df = load_page(order_item)
    fig = px.scatter(df, x =df['price'], y = df['freight_value'],
                 width=800, height=400)
    fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    )
    st.plotly_chart(fig)
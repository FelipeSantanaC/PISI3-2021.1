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
from sklearn.neighbors import KNeighborsClassifier # Importa o algoritmo de ML classificador
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
#Transformando as string de data e hora em timestamp --------------------------------------------------------------------------------------
review_ds['review_creation_date'] = pd.to_datetime(review_ds['review_creation_date'], format='%Y-%m-%d %H:%M:%S')
review_ds['review_answer_timestamp'] = pd.to_datetime(review_ds['review_answer_timestamp'], format='%Y-%m-%d')
orders_ds['order_purchase_timestamp'] = pd.to_datetime(orders_ds['order_purchase_timestamp'], format='%Y-%m-%d %H:%M:%S') 
orders_ds['order_approved_at'] = pd.to_datetime(orders_ds['order_approved_at'], format='%Y-%m-%d %H:%M:%S')
orders_ds['order_delivered_carrier_date'] = pd.to_datetime(orders_ds['order_delivered_carrier_date'], format='%Y-%m-%d %H:%M:%S')
orders_ds['order_delivered_customer_date'] = pd.to_datetime(orders_ds['order_delivered_customer_date'], format='%Y-%m-%d %H:%M:%S')
orders_ds['order_estimated_delivery_date'] = pd.to_datetime(orders_ds['order_estimated_delivery_date'], format='%Y-%m-%d %H:%M:%S')
items_ds['shipping_limit_date'] = pd.to_datetime(items_ds['shipping_limit_date'], format='%Y-%m-%d %H:%M:%S')
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
#Removendo atributos irrelevantes
p3_ds = p3_ds.drop(['order_id','review_id','customer_id','review_comment_title','review_comment_message'],axis=1) #Remove as colunas irrelevante
#Gerando dummy features com os tipos de pagamentos
p3_ds = pd.get_dummies(p3_ds, columns=['payment_type']) #Gera o dummies
#Transformando as variaveis as categorica nominal "delivered" e "canceled" do atributo "order_status" em numericos "1" e "0" 
enconde_order_status = LabelEncoder() #Chama o label responsavel 
labelstatus = enconde_order_status.fit_transform(p3_ds['order_status']) #Transforma em numericos
p3_ds['order_status'] = labelstatus #Atribui os novos valores
#Criando novos atributos com o intervalo entre o timestamp relatado pelo sistema e da entrega ao/pelo cliente
#Valor = data final - data inicial
p3_ds['order_approval_time'] = p3_ds['order_approved_at'] -  p3_ds['order_purchase_timestamp'] #Intervalo entre a compra e sua aprovação
p3_ds['order_picking_time'] = p3_ds['order_delivered_carrier_date'] - p3_ds['order_approved_at'] #Intervalo entre a aprovação e da coleta do pedido
p3_ds['order_delivery_time'] = p3_ds['order_delivered_customer_date'] - p3_ds['order_delivered_carrier_date'] 
p3_ds['delivered_on_time'] = p3_ds['order_estimated_delivery_date'] - p3_ds['order_delivered_customer_date'] #Intervalo entre o tempo estimado e o real da entrega
p3_ds['posted_on_deadline'] = p3_ds['shipping_limit_date'] - p3_ds['order_delivered_carrier_date'] 
p3_ds['review_response_time'] = p3_ds['review_answer_timestamp'] - p3_ds['review_creation_date']
#Transformando os valores das novas colunas em valores numericos(horas)
p3_ds['order_approval_time'] = p3_ds['order_approval_time']/np.timedelta64(1,'h')
p3_ds['order_picking_time'] = p3_ds['order_picking_time']/np.timedelta64(1,'h')
p3_ds['order_delivery_time'] = p3_ds['order_delivery_time']/np.timedelta64(1,'h')
p3_ds['delivered_on_time'] = p3_ds['delivered_on_time']/np.timedelta64(1,'h')
p3_ds['posted_on_deadline'] = p3_ds['posted_on_deadline']/np.timedelta64(1,'h')
p3_ds['review_response_time'] = p3_ds['review_response_time']/np.timedelta64(1,'h')
#Remove os atributos que não serão mais úteis
p3_ds = p3_ds.drop(['review_creation_date',
                'review_answer_timestamp',
                'order_purchase_timestamp',
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date',
                'shipping_limit_date'], axis=1)
#seleciona o sequencial de pagamento = 1 e exclui a respectiva coluna
p3_ds = p3_ds[p3_ds['payment_sequential']==1]
p3_ds = p3_ds.drop(['payment_sequential'],axis=1)
#Dividindo o dataset em treinos e testes 
#Colunas de variaveis 
X = p3_ds[['payment_installments','payment_value','order_status','quantity_of_items',	
            'price','freight_value','payment_type_boleto','payment_type_credit_card','payment_type_debit_card',
            'payment_type_voucher','order_approval_time','order_picking_time','order_delivery_time',
            'delivered_on_time','posted_on_deadline','review_response_time']]
#Colunas com classes
y = p3_ds['review_score'] 
#Configurando conjuntos de teste e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,stratify=y, random_state=999) #Limita o dataset a 20%
#Concatenando os dados de treinamento novamente
X = pd.concat([X_train, y_train], axis=1)
#Separando as classes minoritárias das majoritárias
score5 = X[X['review_score']==5]
score4 = X[X['review_score']==4]
score3 = X[X['review_score']==3]
score2 = X[X['review_score']==2]
score1 = X[X['review_score']==1]
#Reamostrando as classes com menor instâncias até o valor da classe predominante
upsampled1 = resample(score1,replace=True, #Reposição da amostra
                      n_samples=len(score5), #Número de correspondência na classe majoritária
                      random_state=20) #Número de resultados reproduzíveis
upsampled2= resample(score2,replace=True, n_samples=len(score5), random_state=20)
upsampled3 = resample(score3,replace=True, n_samples=len(score5), random_state=20)
upsampled4 = resample(score4,replace=True, n_samples=len(score5), random_state=20)
#Junção de todas as classes
upsampled = pd.concat([upsampled1, upsampled2, upsampled3, upsampled4, score5])
#Entre esse e o original o "y_train" possui uma diferença de 200 instancias
#Separando novamente as classes e varaveis
y_train = upsampled['review_score']
X_train = upsampled[['payment_installments','payment_value','order_status','quantity_of_items',
                    'price','freight_value','payment_type_boleto','payment_type_credit_card','payment_type_debit_card',
                    'payment_type_voucher',	'order_approval_time','order_picking_time','order_delivery_time',
                    'delivered_on_time','posted_on_deadline','review_response_time']]
X_col = X_train.columns
scaler = StandardScaler() #Instancia a função responsavel
X_train = scaler.fit_transform(X_train) #Transformação do "X_train" para o grafico
X_train = pd.DataFrame(X_train,columns=X_col) #Transporma o "X_train" novamente em Dataframe e define o título das colunas
#FUNÇÕES QUE PLOTAM OS GRAFICOS DE ANALISEs====================================================================================================
#PERGUNTA 1 ---------------------------------------------------------------------------------------------------------------------------------
def grafico1_p1():
  fig = px.box(prod_p1['product_description_lenght'], height = 500 , width=800)
  st.plotly_chart(fig)
#PERGUNTA 2 ---------------------------------------------------------------------------------------------------------------------------------
def grafico1_p2():
  fig = px.histogram(payment_ds, x="payment_value", nbins=100)
  st.plotly_chart(fig)
def grafico3_p2():
  fig = px.box(pd.melt(p2_ds),x='variable',y='value',points='outliers')
  st.plotly_chart(fig)
#PERGUNTA 3 ---------------------------------------------------------------------------------------------------------------------------------
def grafico3_p3(): 
  corr = X_train.corr()
  fig = px.imshow(corr,text_auto=True, labels=dict(x='variables',y="Variables", aspect="auto"),
                  x=X_train.columns,
                  y=X_train.columns)
  fig.update_xaxes(
      showticklabels = False
  )
  st.plotly_chart(fig)
#MINERAÇÃO DE DADOS --------------------------------------------------------------------------------------------------------------------------
def grafico_elbow():
  elbow_df = pd.DataFrame({'Clusters': K, 'within-clusters sum-of-squares': distortions})
  fig = (px.line(elbow_df, x='Clusters', y='within-clusters sum-of-squares', template='seaborn')).update_traces(mode='lines+markers')
  st.plotly_chart(fig)
def grafico_matriz_dispersao():
  fig = px.scatter_matrix(p2_ds, dimensions=['payment_value',	'recency',	'frequency'], color="k-means")
  st.plotly_chart(fig)
def grafico_knn(): 
  fig = px.bar(x=X_train.columns,y=importance)
  fig.update_xaxes(
   tickangle=45
  )
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
#ANALISE EXPLORATÓRIA-------------------------------------------------------------------------------------------------------------------
#CASO----------------------------------------------------------------------------------------
if select_page == 'Análise exploratória de dados': #Pagina de Analise exploratória
  st.title('Análise exploratória de dados')
  #Pergunta 1 -------------------------------------------------------------------------------------
  st.subheader('Pergunta 1')
  st.subheader('Caracteres na descrição dos produtos')
  grafico1_p1()
  #Pergunta 2 -------------------------------------------------------------------------------------
  st.subheader('Pergunta 2')
  st.subheader('Frequência de aparições de valores de a cada 100 reais')
  grafico1_p2()
  st.subheader('Correlação de variaveis')
  st.subheader('Verificar outliers')
  grafico2_p2()
  grafico3_p2()
  customer_ds.value_counts('frequency')
  #Pergunta 3 -------------------------------------------------------------------------------------
  st.subheader('Matriz de correlação')
  grafico3_p3()
#DATA MINING ---------------------------------------------------------------------------------------------------------------------------
elif select_page == 'Mineração de dados':
  st.title('Mineração de dados')
  st.subheader('K-Means')
  #KMEANS ------------------------------------------------------------------------------------------------------------------------------
  #Calculando a soma dos quadrados intra-clusters 
  distortions = [] # Representa o 'within-clusters sum-of-squares'
  K = range(1,10) #Representa a quantidade de clusters
  for k in K:
      kmeanModel = KMeans(n_clusters=k)
      kmeanModel.fit(p2_ds)
      distortions.append(kmeanModel.inertia_)
  #Criando nova coluna para o uso do K-means
  newcolumn = pd.DataFrame(kmeanModel.labels_) #Cria uma coluna para gerar rotulos de cores para o grafico
  p2_ds['k-means']=newcolumn #Adiciona a nova coluna ao "pd"
  #Construção do grafico da matriz de dispersão
  grafico_matriz_dispersao()
  #KNN-----------------------------------------------------------------------------------------------
  st.subheader('K-NN')
  st.subheader('Importância dos Atributos')
  #instanciando classificador
  KNNclf = KNeighborsClassifier() 
  #treinando o modelo 
  KNNclf.fit(X_train, y_train)  
  #permutando os valores das variaveis 
  results = permutation_importance(KNNclf, X_train, y_train, scoring='accuracy',random_state=45)
  importance = results.importances_mean
  grafico_knn()

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

#IMPORTAÇÃO DAS BIBLIOTECAS===================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import datetime as dt
#Bibliotecas de classes de machine learning 
from sklearn.cluster import KMeans #Importa o Kmenas 
from sklearn.model_selection import train_test_split  #Divide as instancias entre treino e teste
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Transforma variaveis categoricas em numericas
from sklearn.neighbors import KNeighborsClassifier # Importa o algoritmo de ML classificador
from sklearn.preprocessing import StandardScaler #Importa o algitmo "padronizador"
from sklearn.utils import resample #Reamostragem
#ML
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, precision_score, recall_score, plot_precision_recall_curve, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform, expon
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
st.set_option('deprecation.showPyplotGlobalUse', False)
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
#p2_ds = pd.DataFrame.rename(p2_ds, columns={0:'frequency', 1:'payment_value',	2:'recency'}) #retorna o nome das colunas
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
#cria uma copia com todos as classes target
p3_copy = p3_ds.copy()
p3_copy = p3_copy.sort_values('review_score', ascending=True)
#Dividindo o dataset em treinos e testes 
values=[1,5] #adicionar aqui pega as classes de valor 1,5
p3_ds = p3_ds[p3_ds.review_score.isin(values)] # seleciona especificamente os valores anteriores 
#Colunas de variaveis 
X = p3_ds[['payment_installments','payment_value','order_status','quantity_of_items',	
            'price','freight_value','payment_type_boleto','payment_type_credit_card','payment_type_debit_card',
            'payment_type_voucher','order_approval_time','order_picking_time','order_delivery_time',
            'delivered_on_time','posted_on_deadline','review_response_time']]
#Colunas com classes
y = p3_ds['review_score'] 
#Configurando conjuntos de teste e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,stratify=y, random_state=999) #Limita o dataset a 20%
#Concatenando os dados de treinamento novamente
X = pd.concat([X_train, y_train], axis=1)
#Separando as classes minoritárias das majoritárias
score5 = X[X['review_score']==5]
score1 = X[X['review_score']==1]
#Reamostrando as classes com menor instâncias até o valor da classe predominante
upsampled1 = resample(score1,replace=True, #Reposição da amostra
                      n_samples=len(score5), #Número de correspondência na classe majoritária
                      random_state=20) #Número de resultados reproduzíveis

#Junção de todas as classes
upsampled = pd.concat([upsampled1, score5])
#Entre esse e o original o "y_train" possui uma diferença de 200 instancias
#Separando novamente as classes e varaveis
y_train = upsampled['review_score']
X_train = upsampled[['payment_installments','payment_value','order_status','quantity_of_items',
                    'price','freight_value','payment_type_boleto','payment_type_credit_card','payment_type_debit_card',
                    'payment_type_voucher',	'order_approval_time','order_picking_time','order_delivery_time',
                    'delivered_on_time','posted_on_deadline','review_response_time']]
X_col = X_train.columns
padronizador = StandardScaler() #Instancia a função responsavel
#FUNÇÕES QUE PLOTAM OS GRAFICOS DE ANALISEs====================================================================================================

#PERGUNTA 1 ---------------------------------------------------------------------------------------------------------------------------------
def grafico1_p1():
  fig = px.box(prod_p1['product_description_lenght'], height = 500 , width=800)
  st.plotly_chart(fig)
def grafico2_p1():
  fig = px.scatter(prod_p1, x='product_description_lenght', y='sold_count',size='product_photos_qty', color='product_category_name')
  st.plotly_chart(fig)
#PERGUNTA 2 ---------------------------------------------------------------------------------------------------------------------------------
def grafico1_p2():
  fig = px.histogram(payment_ds, x="payment_value", nbins=100)
  st.plotly_chart(fig)
def grafico2_p2():
  corr = p2_ds.corr()
  fig = px.imshow(corr,text_auto=True, labels=dict(x="Variables", y="Variables"),
                  x=['frequency','payment_value','recency'],
                  y=['frequency','payment_value','recency']
                )
  st.plotly_chart(fig)
def grafico3_p2():
  fig = px.box(pd.melt(p2_ds),x='variable',y='value',points='outliers')
  st.plotly_chart(fig)
#PERGUNTA 3 ---------------------------------------------------------------------------------------------------------------------------------
def grafico2_p3():
  st.write('Este gráfico mostra a dispersão de valores de pagamento e frete, os pontos de dados estão coloridos de acordo com uma escala de valores contínuos da variável delivered_on_time. Esse atributo foi obtido a partir da diferença entre a data de entrega estimada e a data que foi entregue, assim, valores negativos representam atrasos(horas), e os positivos, pedidos que foram entregues antes do prazo.')
  fig = px.scatter(p3_copy, x='freight_value',y='payment_value', animation_frame='review_score',color='delivered_on_time')
  st.plotly_chart(fig)
def grafico1_p3():
  fig = px.histogram(review_ds, x="review_score", color='review_score')
  st.plotly_chart(fig)
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
# STREAMLIT VISUALIZATION=====================================================================================================================
st.title("Cosmus - Visualização de Dados")
#cria barra lateral
st.sidebar.title("Menu")
#cria uma selectbox para navegar entre as páginas de análises
select_page = st.sidebar.selectbox("Selecionar sessão",['Análise exploratória de dados','Mineração de dados',])
#ANALISE EXPLORATÓRIA-------------------------------------------------------------------------------------------------------------------
#CASO----------------------------------------------------------------------------------------
if select_page == 'Análise exploratória de dados': #Pagina de Analise exploratória
  st.title('Análise exploratória de dados')
  #Pergunta 1 -------------------------------------------------------------------------------------
  st.subheader('As quantidade de informações disponíveis na descrição dos produtos têm relevância na sua quantidade de vendas?')
  st.subheader('Caracteres na descrição dos produtos')
  grafico1_p1()
  st.subheader('Relação da descrição do produto com a quantidade de vendas')
  prod_p1 = prod_p1.loc[:5000] #Selecionando os 5000 primeiros produtos com maiores caracteres.
  grafico2_p1()
  #Pergunta 2 -------------------------------------------------------------------------------------
  st.subheader('Quantos grupos de clientes é possível identificar?')
  st.subheader('Distribuição de valores de pagamento a cada R$100')
  grafico1_p2()
  st.subheader('Correlação de variaveis')
  grafico2_p2()
  st.subheader('Variação dos dados')
  grafico3_p2()
  customer_ds.value_counts('frequency')
  #Pergunta 3 -------------------------------------------------------------------------------------
  st.subheader('Quais atributos têm maior peso nas avaliações dos usuários?')
  st.subheader('Dispersão de valores de pagamento e frete ')
  grafico2_p3()
  st.subheader('Distribuição de Review Score')
  grafico1_p3()
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
  st.subheader('Grafico do Elbow Method')
  #Costrução do gráfico que mostra o resultado do elbow method. 
  grafico_elbow()
   #Informando ao K-means a quantidade ideal de clusters.
  st.subheader('Matriz de Dispersão')
  kmeanModel = KMeans(n_clusters=4)
  kmeanModel.fit(p2_ds)
  newcolumn = pd.DataFrame(kmeanModel.labels_)
  p2_ds = pd.DataFrame(p2_ds)
  p2_ds = pd.DataFrame.rename(p2_ds, columns={0:'payment_value',	1:'recency',	2:'frequency'})
  p2_ds['k-means']=newcolumn
  grafico_matriz_dispersao()
  #AQUI COMEÇA O APRENDIZADO BOY
  def model_selection():
  #Rodando modelo KNN==========================================KNN===============================================================
    st.title("Avaliação de Modelos")
    st.subheader("KNN")
    KNNclf = KNeighborsClassifier()      
    hiperparametros_KNN =[{'classificador__n_neighbors': randint (1, 50)}]
    pipe = Pipeline(steps=[('padronizador', padronizador),('classificador', KNNclf)])
    melhor_modelo_KNN = RandomizedSearchCV(pipe, param_distributions=hiperparametros_KNN, cv=10, n_jobs=-1)
    melhor_modelo_KNN.fit(X_train, y_train)
    #Avaliando modelo KNN
    y_pred_KNN = melhor_modelo_KNN.predict(X_test)
    #Matriz de confusão
    labels = ['1','5']
    plot_confusion_matrix(melhor_modelo_KNN, X_test, y_test, display_labels=labels)
    st.pyplot()
    #Métricas
    acuracia = accuracy_score(y_test, y_pred_KNN)
    precisao = precision_score(y_test, y_pred_KNN,average='macro')
    cobertura = recall_score(y_test, y_pred_KNN,average='macro')
    F1_Score = f1_score(y_test, y_pred_KNN,average='macro')
    st.text('KNN: Acuracia:%.3f , Precisao: %.3f , Recall: %.3f , F1Score: %.3f'%(acuracia, precisao, cobertura, F1_Score))
    #gráfico curva roc KNN
  # Coletando predições
    y_prob = melhor_modelo_KNN.predict_proba(X_test)[:, 1] # probabilidades para a 2a classe
    curva_precisao, curva_cobertura, thresholds = roc_curve(y_test, y_prob,pos_label=5 )
    fig = px.area(
      x=curva_precisao, y=curva_cobertura,
      title=f'ROC Curve KNN (AUC={auc(curva_precisao, curva_cobertura):.4f})',
      labels=dict(x='Taxa de falsos positivos', y='Taxa de verdadeiros positivos'),
      width=700, height=500
    )
    fig.add_shape(
      type='line', line=dict(dash='dash'),
      x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)
    #==============================================================================================================================================
    #Rodando modelo de decision tree=========================================DECISION TREE=========================================================
    st.subheader("Decision Tree")
    DTclf = DecisionTreeClassifier()
    hiperparametros_DT = [{'classificador__min_samples_split':  uniform(loc=1e-6, scale=0.5), 
                                  'classificador__min_samples_leaf':  uniform(loc=1e-6, scale=0.5),
                                  'classificador__max_depth':  randint(1, 1000),
                                  'classificador__criterion':  ['entropy', 'gini']}]
    pipe = Pipeline(steps=[('padronizador', padronizador),('classificador', DTclf)])
    melhor_modelo_DT = RandomizedSearchCV(pipe, param_distributions=hiperparametros_DT, cv=10)
    melhor_modelo_DT.fit(X_train, y_train)
    #Avaliando modelo decision tree
    y_pred_DT = melhor_modelo_DT.predict(X_test)
    #matriz de confusão decision tree
    plot_confusion_matrix(melhor_modelo_DT, X_test, y_test, display_labels=labels)
    st.pyplot()
    #metricas
    acuracia = accuracy_score(y_test, y_pred_DT)
    precisao = precision_score(y_test, y_pred_DT,average='macro')
    cobertura = recall_score(y_test, y_pred_DT,average='macro')
    F1_Score = f1_score(y_test, y_pred_DT,average='macro')
    print('Decision Tree: Acuracia:%.3f , Precisao: %.3f , Recall: %.3f , F1Score: %.3f'%(acuracia, precisao, cobertura, F1_Score))
    #gráfico curva roc DECISION TREE
    # Coletando predições
    y_prob = melhor_modelo_DT.predict_proba(X_test)[:, 1] # probabilidades para a 2a classe
    curva_precisao, curva_cobertura, thresholds = roc_curve(y_test, y_prob,pos_label=5 )
    fig = px.area(
        x=curva_precisao, y=curva_cobertura,
        title=f'ROC Curve Decision Tree (AUC={auc(curva_precisao, curva_cobertura):.4f})',
        labels=dict(x='Taxa de falsos positivos', y='Taxa de verdadeiros positivos'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)
    #========================================================================================================================================================
    #rodando naive bayes!!=====================================================GAUSSIAN_NB===================================================================
    st.subheader("Naive Bayes")
    NBclf = GaussianNB()
    parametros_NB = [{'classificador__var_smoothing': np.logspace(0,-9, num=100)}]
    pipe = Pipeline(steps=[('padronizador', padronizador),('classificador', NBclf)])
    melhor_modelo_NB = RandomizedSearchCV(pipe, param_distributions=parametros_NB, cv=10)
    melhor_modelo_NB.fit(X_train, y_train)
    #Avaliando modelo naive bayes
    y_pred_NB = melhor_modelo_NB.predict(X_test)
    #matriz de confusÃ£o decision tree
    plot_confusion_matrix(melhor_modelo_NB, X_test, y_test, display_labels=labels)
    st.pyplot()
    #metricas
    acuracia = accuracy_score(y_test, y_pred_NB)
    precisao = precision_score(y_test, y_pred_NB,average='macro')
    cobertura = recall_score(y_test, y_pred_NB,average='macro')
    F1_Score = f1_score(y_test, y_pred_NB,average='macro')
    print('GaussianNB: Acuracia:%.3f , Precisao: %.3f , Recall: %.3f , F1Score: %.3f'%(acuracia, precisao, cobertura, F1_Score))
    # curva roc naive bayes
    y_prob = melhor_modelo_NB.predict_proba(X_test)[:, 1] # probabilidades para a 2a classe
    curva_precisao, curva_cobertura, thresholds = roc_curve(y_test, y_prob,pos_label=5 )
    fig = px.area(
        x=curva_precisao, y=curva_cobertura,
        title=f'ROC Curve Naive Bayes(AUC={auc(curva_precisao, curva_cobertura):.4f})',
        labels=dict(x='Taxa de falsos positivos', y='Taxa de verdadeiros positivos'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)
    #=================================================================================================================================================
    #Importância de variáveis com PCA
    st.title("Importancia de atributos")
    X_train_scaled = padronizador.fit_transform(X_train)
    pca = PCA().fit(X_train_scaled)
    fig = px.line(pca.explained_variance_ratio_.cumsum(), 
                  title="Variância explicada por número de componentes principais",
                  labels=dict(index="Número de componentes principais",value="Variância"))
    st.plotly_chart(fig)
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
        index=X_train.columns
    )
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
        index=X_train.columns
    )
    pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']
    
    fig = px.bar(x=pc1_loadings['Attribute'], 
                y=pc1_loadings['CorrelationWithPC1'], 
                title='Importância de variáveis',
                labels=dict(x="Variáveis",y="Correlação com componente principal"))
    st.plotly_chart(fig)
  model_selection()
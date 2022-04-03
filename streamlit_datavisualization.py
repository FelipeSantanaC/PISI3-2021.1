#<<<<<<< HEAD
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid 
import plotly.express as px
#o dataset possui varios arquivos com finalidades divergentes, aqui cada variavel corresponde ao entereço do dataset no repositorio do grupo
customer = "https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_customers_dataset.csv?raw=true" 
geolocation = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_geolocation_dataset.csv?raw=true'
order_item = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_order_items_dataset.csv?raw=true'
order_payment = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_order_payments_dataset.csv?raw=true'
order_review = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_order_reviews_dataset.csv?raw=true'
order = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_orders_dataset.csv?raw=true'
product = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_products_dataset.csv?raw=true'
sellers = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/olist_sellers_dataset.csv?raw=true'
translation = 'https://github.com/FelipeSantanaC/PISI3-2021.1/blob/main/data/product_category_name_translation.csv?raw=true'
#aqui importa o dataset e transforma em um DataFrame para as futuras analises
#cust = pd.read_csv(customer)
#geo = pd.read_csv(geolocation)
#ordI = pd.read_csv(order_item)
#ordP = pd.read_csv(order_payment)
#ordR = pd.read_csv(order_review)
#ord = pd.read_csv(order)
#prod = pd.read_csv(product)
#sell = pd.read_csv(sellers)
#tran = pd.read_csv(translation)
#título que será mostrado em todas as páginas
st.title("Cosmus - Data visualization App")
#cria barra lateral
st.sidebar.title("Menu")
#cria uma selectbox para navegar entre as páginas de análises
select_page = st.sidebar.selectbox("Select dataset",['Order Payments','Sellers list','Products list','Order reviews'])
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
    
    

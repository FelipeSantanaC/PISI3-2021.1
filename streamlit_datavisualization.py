import pandas as pd
import streamlit as st
from st_aggrid import AgGrid 
import plotly.express as px
#carrega os datasets numa lista
data_url = ['https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_order_payments_dataset.csv',
'https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_sellers_dataset.csv',
'https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_products_dataset.csv',
'https://raw.githubusercontent.com/FelipeSantanaC/PISI3-2021.1/main/data/olist_order_reviews_dataset.csv']
#título que será mostrado em todas as páginas
st.title("Cosmus - Data visualization App")
#cria barra lateral
st.sidebar.title("Menu")
#cria uma selectbox para navegar entre as páginas de análises
select_page = st.sidebar.selectbox("Select dataset",['Order Payments','Sellers list','Products list','Order reviews'])
#função que carrega o template de cada página de análise, recebe a lista de datasets e o indice correspondente
def load_page(data,i):
    st.subheader(select_page) #subtítulo
    df = pd.read_csv(data[i]) #leitura do dataset
    AgGrid(df)  #cria dataframe interativo(filtros etc.)
    return df
#os if statements abaixo referem-se as opções do selectbox
if select_page == 'Order Payments':
    df = load_page(data_url,0) # chama funçao de template
    pt_count = df['payment_type'].value_counts() #retorna frequencia de valores
    fig = px.bar(pt_count) #plotagem do gráfico
    st.plotly_chart(fig, use_container_width=True)
#repete a mesma estrutura da condição anterior
elif select_page =='Sellers list':
    df = load_page(data_url,1)
    pt_count = df['seller_state'].value_counts()
    fig = px.bar(pt_count)
    st.plotly_chart(fig)
# cria mais um selectbox para selecionar uma categoria de produto, e plota a distribuição de pesos(g) dos produtos num scatterplot
elif select_page =='Products list':
    df = load_page(data_url,2)
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
    df = load_page(data_url,3)
    score_count = df['review_score'].value_counts()
    fig2 = px.bar(score_count, width=800, height=500)
    st.plotly_chart(fig2)
    
    

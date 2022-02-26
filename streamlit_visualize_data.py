import pandas as pd
import streamlit as st
from st_aggrid import AgGrid 
import plotly.express as px
data_url = ['https://raw.githubusercontent.com/AurileneBagagi/E-commerce_dataset/main/olist_order_payments_dataset.csv',
'https://raw.githubusercontent.com/AurileneBagagi/E-commerce_dataset/main/olist_sellers_dataset.csv',
'https://raw.githubusercontent.com/AurileneBagagi/E-commerce_dataset/main/olist_products_dataset.csv']

st.title("Data visualization App")

st.sidebar.title("Menu")

select_page = st.sidebar.selectbox("Select dataset",['Order Payments','Sellers list','Products list'])

def load_page(data,i):
    st.subheader(select_page)
    df = pd.read_csv(data[i])
    AgGrid(df)  
    return df

if select_page == 'Order Payments':
    df = load_page(data_url,0)
    pt_count = df['payment_type'].value_counts()
    st.bar_chart(pt_count)

elif select_page =='Sellers list':
    df = load_page(data_url,1)
    pt_count = df['seller_state'].value_counts()
    st.bar_chart(pt_count)

elif select_page =='Products list':
    df = load_page(data_url,2)
    category_filter = st.selectbox('Select product category',['cds_dvs_musicais','casa_conforto_2','moveis_quarto','fashion_roupa_infanto_juvenil', 'livros_importados', 'alimentos_bebidas', 'construcao_ferramentas_jardim', 'artigos_de_festa', 'telefonia_fixa', 'fashion_underwear_e_moda_praia', 'climatizacao', 'eletroportateis', 'casa_construcao', 'telefonia', 'moveis_escritorio', 'informatica_acessorios', 'eletrodomesticos', 'utilidades_domesticas', 'perfumaria' ])
    category_selected = df[df['product_category_name']==category_filter]
    fig = px.scatter(category_selected, y="product_category_name", x="product_weight_g")
    st.plotly_chart(fig, use_container_width=True)
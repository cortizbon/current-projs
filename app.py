import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import pywaffle as pwf

DIC_COLORES = {'verde':["#009966"],
               'ro_am_na':["#FFE9C5", "#F7B261","#D8841C", "#dd722a","#C24C31", "#BC3B26"],
               'az_verd': ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
               'ax_viol': ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
               'ofiscal': ["#F9F9F9", "#2635bf"]}

df= pd.read_csv('df_tot.csv')
df['Apropiación en precios corrientes'] /= 1000000000
sectors = df['Sector'].unique()
entities = df['Entidad'].unique()
units = df['Unidad'].unique()
tipo_gastos = df['Tipo de gasto'].unique()
cuentas = df['Cuenta'].dropna().unique()
subcuentas = df['Subcuenta'].dropna().unique()
projects = df['Objeto/proyecto'].dropna().unique()

tab1, tab2, tab3 = st.tabs(['PEPE desagregado', 'Ingreso', 'Territorial'])

with tab1:
    st.header("Treemap")     

    year = st.selectbox("Seleccione el año", 
                     [2019, 2024])
    filter_year = df[df['Año'] == year]
    try:

        fig = px.treemap(filter_year, 
                        path=[px.Constant('PGN'), 'Sector', 
                                'Entidad', 
                                'Unidad',
                                'Tipo de gasto', 'Cuenta', 'Subcuenta'],
                        values='Apropiación en precios corrientes',
                        color='Sector',
                        title="Matriz de composición anual del PGN <br><sup>Cifras en miles de millones de pesos</sup>")
        
        fig.update_layout(width=1000, height=600)
        
        st.plotly_chart(fig)
    except:
        st.warning("Not enough information.")

with tab2:
    geih = pd.read_csv('geih_milciles.csv')
    income = st.number_input("¿De cuánto es su ingreso al mes?", min_value=10000)
    geih.columns = geih.columns.str.strip()
    def funcion(row, valor):
        if row['mean(HHincome)'] < valor:
            return 'menor'
        else:
            return 'mayor'

    geih['pos'] = geih.apply(funcion, args=(income,), axis=1)
    grupo = geih.groupby('pos')['of HHincome'].count().reset_index()

    grupo['monto_promedio'] = geih.groupby('pos')['e(y| y > z)'].first()['mayor'], geih.groupby('pos')['e(y| y <z)'].last()['menor'] 

    fig =plt.figure(
        FigureClass=pwf.Waffle,
        rows=20,
        values=grupo['of HHincome'],
        labels=list(grupo['pos']),  # Legend would not be created without this line
        legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
        icons='user',
        block_aspect_ratio=0.7
    )   
    st.pyplot(fig)




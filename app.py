import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import pywaffle as pwf
from io import BytesIO

DIC_COLORES = {'verde':["#009966"],
               'ro_am_na':["#FFE9C5", "#F7B261","#D8841C", "#dd722a","#C24C31", "#BC3B26"],
               'az_verd': ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
               'ax_viol': ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
               'ofiscal': ["#F9F9F9", "#2635bf"]}
st.set_page_config(layout='wide')
st.title('PePE desagregado')


df= pd.read_csv('df_tot.csv')
df['Apropiación en precios corrientes (cifras en miles en millones de pesos)'] = (df['Apropiación en precios corrientes'] /  1000_000_000).round(2)
sectors = df['Sector'].unique()
entities = df['Entidad'].unique()
units = df['Unidad'].unique()
tipo_gastos = df['Tipo de gasto'].unique()
cuentas = df['Cuenta'].dropna().unique()
subcuentas = df['Subcuenta'].dropna().unique()
projects = df['Objeto/proyecto'].dropna().unique()

tab1, tab2, tab3 = st.tabs(['PEPE desagregado', 'Treemap', 'Descarga de datos'])

with tab1:
    # cambio porcentual general

    col1, col2, col3 = st.columns(3)
    tot_2024 = df[df['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = df[df['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # cambio porcentual por sector

    sector = st.selectbox("Seleccione un sector: ", sectors)

    f_s = df[df['Sector'] == sector]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s[f_s['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s[f_s['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {sector} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {sector} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric(f"Variación anualizada {sector}", f"{round(change * 100, 2)}%")

    # cambio porcentual por entidad
    entidad = st.selectbox("Seleccione una entidad: ", f_s['Entidad'].unique())

    f_s_e = f_s[f_s['Entidad'] == entidad]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e[f_s_e['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e[f_s_e['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {entidad} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {entidad} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada ", f"{round(change * 100, 2)}%")

    # cambio porcentual por unidad

    
    unidad = st.selectbox("Seleccione una unidad: ", f_s_e['Unidad'].unique())

    f_s_e_u = f_s_e[f_s_e['Unidad'] == unidad]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_u[f_s_e_u['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_u[f_s_e_u['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {unidad} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {unidad} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")
    # cambio porcentual por tipo de gasto

    tipo_gasto = st.selectbox("Seleccione un tipo de gasto: ", f_s_e_u['Tipo de gasto'].unique())

    f_s_e_u_tg = f_s_e_u[f_s_e_u['Tipo de gasto'] == tipo_gasto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_u_tg[f_s_e_u_tg['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_u_tg[f_s_e_u_tg['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {tipo_gasto} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {tipo_gasto} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")
    

    # cambio porcentual por cuenta

    cuenta = st.selectbox("Seleccione una cuenta: ", f_s_e_u_tg['Cuenta'].unique())

    f_s_e_u_tg_c = f_s_e_u_tg[f_s_e_u_tg['Cuenta'] == cuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_u_tg_c[f_s_e_u_tg_c['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_u_tg_c[f_s_e_u_tg_c['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {cuenta} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {cuenta} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # subcuentas
    
    subcuenta = st.selectbox("Seleccione una subcuenta: ", f_s_e_u_tg_c['Subcuenta'].unique())

    f_s_e_u_tg_c_sc = f_s_e_u_tg_c[f_s_e_u_tg_c['Subcuenta'] == subcuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_u_tg_c_sc[f_s_e_u_tg_c_sc['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_u_tg_c_sc[f_s_e_u_tg_c_sc['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {subcuenta} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {subcuenta} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # objeto
        
    objeto = st.selectbox("Seleccione un objeto/proyecto: ", f_s_e_u_tg_c_sc['Objeto/proyecto'].unique())

    f_s_e_u_tg_c_sc_o = f_s_e_u_tg_c_sc[f_s_e_u_tg_c_sc['Objeto/proyecto'] == objeto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_u_tg_c_sc_o[f_s_e_u_tg_c_sc_o['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_u_tg_c_sc_o[f_s_e_u_tg_c_sc_o['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {objeto} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {objeto} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")    
        
with tab2:
    st.header("Treemap")     

    year = st.selectbox("Seleccione el año", 
                     [2019, 2024])
    filter_year = df[df['Año'] == year]
    


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
    
    st.warning("Not enough information.")


with tab3:
    st.subheader("Descarga de dataset completo")


    binary_output = BytesIO()
    df.to_excel(binary_output, index=False)
    st.download_button(label = 'Descargar datos completos',
                    data = binary_output.getvalue(),
                    file_name = 'datos_desagregados_2019_2024.xlsx')





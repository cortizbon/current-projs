import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import pywaffle as pwf
from io import BytesIO

from utils import create_dataframe_sankey

DIC_COLORES = {'verde':["#009966"],
               'ro_am_na':["#FFE9C5", "#F7B261","#D8841C", "#dd722a","#C24C31", "#BC3B26"],
               'az_verd': ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
               'ax_viol': ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
               'ofiscal': ["#F9F9F9", "#2635bf"]}
st.set_page_config(layout='wide')
st.title('PePE desagregado')

"#F9F9F9" "#FFE9C5"
df = pd.read_csv('datasets/datos_desagregados_2019_2024.csv')
df['Apropiación en precios corrientes (cifras en miles de millones de pesos)'] = (df['Apropiación en precios corrientes'] /  1000_000_000).round(2)

df2 = pd.read_csv('datasets/anteproyecto_2025.csv')
sectors = df['Sector'].unique()
entities = df['Entidad'].unique()
tipo_gastos = df['Tipo de gasto'].unique()
cuentas = df['Cuenta'].dropna().unique()
subcuentas = df['Subcuenta'].dropna().unique()
projects = df['Objeto/proyecto'].dropna().unique()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['PEPE desagregado', 
                                        'Treemap', 
                                        'Descarga de datos', 
                                        'SankeyDiagrams',
                                        "Lollipop",
                                        "Anteproyecto - 2025"])

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


    # cambio porcentual por tipo de gasto

    tipo_gasto = st.selectbox("Seleccione un tipo de gasto: ", f_s_e['Tipo de gasto'].unique())

    f_s_e_tg = f_s_e[f_s_e['Tipo de gasto'] == tipo_gasto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg[f_s_e_tg['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_tg[f_s_e_tg['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {tipo_gasto} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {tipo_gasto} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")
    
    # cambio porcentual por cuenta

    cuenta = st.selectbox("Seleccione una cuenta: ", f_s_e_tg['Cuenta'].unique())

    f_s_e_tg_c = f_s_e_tg[f_s_e_tg['Cuenta'] == cuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c[f_s_e_tg_c['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_tg_c[f_s_e_tg_c['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {cuenta} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {cuenta} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # subcuentas
    
    subcuenta = st.selectbox("Seleccione una subcuenta: ", f_s_e_tg_c['Subcuenta'].unique())

    f_s_e_tg_c_sc = f_s_e_tg_c[f_s_e_tg_c['Subcuenta'] == subcuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {subcuenta} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {subcuenta} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # objeto
        
    objeto = st.selectbox("Seleccione un objeto/proyecto: ", f_s_e_tg_c_sc['Objeto/proyecto'].unique())

    f_s_e_tg_c_sc_o = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Objeto/proyecto'] == objeto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {objeto} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {objeto} | 2024", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")



    sub_proyecto = st.selectbox("Seleccione un subproyecto: ", f_s_e_tg_c_sc_o['Subproyecto'].unique())

    f_s_e_tg_c_sc_o_sp = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Subproyecto'] == sub_proyecto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc_o_sp[f_s_e_tg_c_sc_o_sp['Año'] == 2024]['Apropiación en precios corrientes'].sum()
    tot_2019 = f_s_e_tg_c_sc_o_sp[f_s_e_tg_c_sc_o_sp['Año'] == 2019]['Apropiación en precios corrientes'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {sub_proyecto} | 2019", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {sub_proyecto} | 2024", round(tot_2024 / 1_000_000_000, 2))

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
                                    'Tipo de gasto', 'Cuenta'],
                            values='Apropiación en precios corrientes (cifras en miles de millones de pesos)',
                            title="Matriz de composición anual del PGN <br><sup>Cifras en miles de millones de pesos</sup>",
                            color_continuous_scale='Teal')
            
    fig.update_layout(width=1000, height=600)
            
    st.plotly_chart(fig)

with tab3:
    st.subheader("Descarga de dataset completo")


    binary_output = BytesIO()
    df.to_excel(binary_output, index=False)
    st.download_button(label = 'Descargar datos completos',
                    data = binary_output.getvalue(),
                    file_name = 'datos_desagregados_2019_2024.xlsx')
    
import plotly.graph_objects as go
with tab4:
    fig  = go.Figure(data=go.Sankey(

    ))

with tab5:
    st.header("Cambio 2019 - 2024")
    st.subheader("Cambio por sector")
    data = df.pivot_table(index='Sector',
               columns='Año',
               values='Apropiación en precios corrientes',
               aggfunc='sum').sort_values(by=2024).tail(15).div(1_000_000_000).round(2).reset_index()

    my_range = range(1, len(data['Sector']) + 1)

    trace1 = go.Scatter(
        x=data[2024],
        y=list(my_range),
        mode='markers',
        name='2024',
        hovertext=data['Sector'],
        hoverinfo='text+x',
        marker_color="#2635bf",
        marker_size=10
    )
    trace2 = go.Scatter(
        x=data[2019],
        y=list(my_range),
        mode='markers',
        name='2019',
        hovertext=data['Sector'],
        hoverinfo='text+x',
        marker_color="#D8841C",
        marker_size=10
    )

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)


    fig.update_layout(
        title='Cambio del gasto por sector (2019 - 2024)',
        xaxis_title='Gasto (miles de millones de pesos)',
        yaxis_title='',
        yaxis=dict(tickvals=list(my_range), ticktext=["" for i in my_range]),
        xaxis_tickformat="4.",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=1000, height=600)
    for i, j, k in zip(my_range, data[2019], data[2024]):
        fig.add_shape(type='line', x0=j, x1=k, y0=i, y1=i, line_color="#2635bf", line_width=0.5) 

    st.plotly_chart(fig)

    st.subheader("Cambio por entidad")
    data = df.pivot_table(index='Entidad',
               columns='Año',
               values='Apropiación en precios corrientes',
               aggfunc='sum').sort_values(by=2024).dropna().tail(15).div(1_000_000_000).round(2).reset_index()

    my_range = range(1, len(data['Entidad']) + 1)

    trace1 = go.Scatter(
        x=data[2024],
        y=list(my_range),
        mode='markers',
        name='2024',
        hovertext=data['Entidad'],
        hoverinfo='text+x',
        marker_color="#2635bf",
        marker_size=10
    )
    
    trace2 = go.Scatter(
        x=data[2019],
        y=list(my_range),
        mode='markers',
        name='2019',
        hovertext=data['Entidad'],
        hoverinfo='text+x',
        marker_color="#D8841C",
        marker_size=10
    )

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)


    fig.update_layout(
        title='Cambio del gasto por entidad (2019 - 2024)',
        xaxis_title='Gasto (miles de millones de pesos)',
        yaxis_title='',
        yaxis=dict(tickvals=list(my_range), ticktext=["" for i in my_range]),
        xaxis_tickformat="4.",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=1000, height=600)
    for i, j, k in zip(my_range, data[2019], data[2024]):
        fig.add_shape(type='line', x0=j, x1=k, y0=i, y1=i, line_color="#2635bf", line_width=0.5) 
 

    st.plotly_chart(fig)

with tab6:
    st.header("Anteproyecto")

    st.subheader("Treemap")

    fig = px.treemap(df2, 
                            path=[px.Constant('Anteproyecto'), 'ENTIDAD', 
                                    'CONCEPTO'],
                            values='TOTAL',
                            title="Matriz de composición anual del Anteproyecto <br><sup>Cifras en millones de pesos</sup>",
                            color_continuous_scale='Teal')
            
    fig.update_layout(width=1000, height=600)
            
    st.plotly_chart(fig)

    st.subheader("Flujo del gasto")
    lista = ['ENTIDAD', 'CONCEPTO']

    top_10 = df2.groupby('ENTIDAD')['TOTAL'].sum().reset_index().sort_values(by='TOTAL', ascending=False).head(10)['ENTIDAD']

    top_10_df = df2[df2['ENTIDAD'].isin(top_10)]

    rev_info, conc = create_dataframe_sankey(top_10_df, 'TOTAL',*lista)
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "#2635bf", width = 0.5),
      label = list(rev_info.keys()),
      color = "#2635bf"
    ),
    link = dict(
      source = conc['source'], # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = conc['target'],
      value = conc['value']
    ))])

    fig.update_layout(title_text="Flujo de gasto", font_size=10, width=1000, height=600)
    st.plotly_chart(fig)




    st.subheader("Descarga de datos")


    binary_output = BytesIO()
    df2.to_excel(binary_output, index=False)
    st.download_button(label = 'Descargar datos de anteproyecto',
                    data = binary_output.getvalue(),
                    file_name = 'datos_agregados_anteproyecto_2025.xlsx')











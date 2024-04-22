import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 
import pywaffle as pwf
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import create_dataframe_sankey

DIC_COLORES = {'verde':["#009966"],
               'ro_am_na':["#FFE9C5", "#F7B261","#D8841C", "#dd722a","#C24C31", "#BC3B26"],
               'az_verd': ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
               'ax_viol': ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
               'ofiscal': ["#F9F9F9", "#2635bf"]}
dict_gasto = {'Funcionamiento':DIC_COLORES['az_verd'][2],
              'Deuda':DIC_COLORES['ax_viol'][1],
              'Inversión':DIC_COLORES['ro_am_na'][3]}
st.set_page_config(layout='wide')
st.title('PePE desagregado')

df = pd.read_csv('datasets/datos_desagregados_2019_2024.csv')
df['Apropiación en precios corrientes (cifras en miles de millones de pesos)'] = (df['Apropiación en precios corrientes'] /  1000_000_000).round(2)

test_df = pd.read_csv('datasets/dataset_192425.csv')
df2 = pd.read_csv('datasets/anteproyecto_2025.csv')
sectors = df['Sector'].unique()
entities = df['Entidad'].unique()
tipo_gastos = df['Tipo de gasto'].unique()
cuentas = df['Cuenta'].dropna().unique()
subcuentas = df['Subcuenta'].dropna().unique()
projects = df['Objeto/proyecto'].dropna().unique()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['PEPE desagregado', 
                                        'Treemap', 
                                        'Descarga de datos',
                                        "Lollipop",
                                        "Anteproyecto - 2025",
                                        'Actualización 2025',
                                        'Diccionario',
                                        'Convertir datos'])

with tab1:
    # cambio porcentual general

    col1, col2 = st.columns(2)

    with col1:
        year_base = st.selectbox("Seleccione el año base: ", test_df['Año'].unique(), 0)

    with col2:
        year_vs = st.selectbox("Seleccione el año a comparar: ", test_df['Año'].unique(), 1)
    df = test_df.copy()

    if year_vs < year_base:
        st.warning("El año a comparar debe ser mayor o igual al año base.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    tot_2024 = df[df['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = df[df['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # cambio porcentual por sector

    sector = st.selectbox("Seleccione un sector: ", sectors)

    f_s = df[df['Sector'] == sector]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s[f_s['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s[f_s['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {sector} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {sector} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric(f"Variación anualizada {sector}", f"{round(change * 100, 2)}%")


    # cambio porcentual por entidad
    entidad = st.selectbox("Seleccione una entidad: ", f_s['Entidad'].unique())

    f_s_e = f_s[f_s['Entidad'] == entidad]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e[f_s_e['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e[f_s_e['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (2024 - 2019)))- 1
    with col1:
        st.metric(f"Apropiación | {entidad} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {entidad} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada ", f"{round(change * 100, 2)}%")


    # cambio porcentual por tipo de gasto

    tipo_gasto = st.selectbox("Seleccione un tipo de gasto: ", f_s_e['Tipo de gasto'].unique())

    f_s_e_tg = f_s_e[f_s_e['Tipo de gasto'] == tipo_gasto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg[f_s_e_tg['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e_tg[f_s_e_tg['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {tipo_gasto} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {tipo_gasto} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")
    
    # cambio porcentual por cuenta

    cuenta = st.selectbox("Seleccione una cuenta: ", f_s_e_tg['Cuenta'].unique())

    f_s_e_tg_c = f_s_e_tg[f_s_e_tg['Cuenta'] == cuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c[f_s_e_tg_c['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e_tg_c[f_s_e_tg_c['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {cuenta} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {cuenta} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # subcuentas
    
    subcuenta = st.selectbox("Seleccione una subcuenta: ", f_s_e_tg_c['Subcuenta'].unique())

    f_s_e_tg_c_sc = f_s_e_tg_c[f_s_e_tg_c['Subcuenta'] == subcuenta]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {subcuenta} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {subcuenta} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")

    # objeto
        
    objeto = st.selectbox("Seleccione un objeto/proyecto: ", f_s_e_tg_c_sc['Objeto/proyecto'].unique())

    f_s_e_tg_c_sc_o = f_s_e_tg_c_sc[f_s_e_tg_c_sc['Objeto/proyecto'] == objeto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {objeto} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {objeto} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")



    sub_proyecto = st.selectbox("Seleccione un subproyecto: ", f_s_e_tg_c_sc_o['Subproyecto'].unique())

    f_s_e_tg_c_sc_o_sp = f_s_e_tg_c_sc_o[f_s_e_tg_c_sc_o['Subproyecto'] == sub_proyecto]
    col1, col2, col3 = st.columns(3)
    tot_2024 = f_s_e_tg_c_sc_o_sp[f_s_e_tg_c_sc_o_sp['Año'] == year_vs]['Apropiación en precios constantes (2025)'].sum()
    tot_2019 = f_s_e_tg_c_sc_o_sp[f_s_e_tg_c_sc_o_sp['Año'] == year_base]['Apropiación en precios constantes (2025)'].sum()

    change = ((tot_2024 / tot_2019) ** (1 / (year_vs - year_base)))- 1
    with col1:
        st.metric(f"Apropiación | {sub_proyecto} | {year_base}", round(tot_2019 / 1_000_000_000, 2))

    with col2:
        st.metric(f"Apropiación | {sub_proyecto} | {year_vs}", round(tot_2024 / 1_000_000_000, 2))

    with col3:
        st.metric("Variación anualizada", f"{round(change * 100, 2)}%")    
        
with tab2:
    st.header("Treemap")     

    year = st.selectbox("Seleccione el año", 
                     [2019, 2024, 2025])
    filter_year = df[df['Año'] == year]
    


    fig = px.treemap(filter_year, 
                            path=[px.Constant('PGN'), 'Sector', 
                                    'Entidad',
                                    'Tipo de gasto', 'Cuenta'],
                            values='Apropiación en precios constantes (2025)',
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
    

with tab4:
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

with tab5:
    st.header("Anteproyecto")

    st.subheader("Treemap")

    fig = px.treemap(df2, 
                            path=[px.Constant('Anteproyecto'), 'Sector', 'ENTIDAD','Tipo de gasto', 
                                    'CONCEPTO'],
                            values='TOTAL',
                            title="Matriz de composición anual del Anteproyecto <br><sup>Cifras en millones de pesos</sup>",
                            color_continuous_scale='Teal')
            
    fig.update_layout(width=1000, height=600)
            
    st.plotly_chart(fig)

    st.subheader("Flujo del gasto por entidad")
    lista = ['ENTIDAD', 'Tipo de gasto', 'CONCEPTO']

    top_10 = df2.groupby('ENTIDAD')['TOTAL'].sum().reset_index().sort_values(by='TOTAL', ascending=False).head(10)['ENTIDAD']

    top_10_df = df2[df2['ENTIDAD'].isin(top_10)]

    dicti = {'source':['Inversion','Servicio de la deuda']}
    rev_info, conc = create_dataframe_sankey(top_10_df, 'TOTAL',*lista, **dicti)
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

    st.subheader("Flujo del gasto por sector")
    lista = ['Sector', 'Tipo de gasto', 'CONCEPTO']

    top_10 = df2.groupby('Sector')['TOTAL'].sum().reset_index().sort_values(by='TOTAL', ascending=False).head(10)['Sector']

    top_10_df = df2[df2['Sector'].isin(top_10)]

    dicti = {'source':['Inversion','Servicio de la deuda']}
    rev_info, conc = create_dataframe_sankey(top_10_df, 'TOTAL',*lista, **dicti)
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

    fig.update_layout(title_text="Flujo de gasto por sector", font_size=10, width=1000, height=600)
    st.plotly_chart(fig)






    st.subheader("Descarga de datos")


    binary_output = BytesIO()
    df2.to_excel(binary_output, index=False)
    st.download_button(label = 'Descargar datos de anteproyecto',
                    data = binary_output.getvalue(),
                    file_name = 'datos_agregados_anteproyecto_2025.xlsx')

with tab6:

    data = pd.read_csv('datasets/datos_def_2025.csv')

    st.subheader('General')

    piv_2025 = data.groupby('Año')['Apropiación a precios constantes (2025)'].sum().reset_index()

    #piv_2024['Apropiación a precios constantes (2024)'] /= 1000

    fig = make_subplots(rows=1, cols=2, x_title='Año',  )
    
    fig.add_trace(
        go.Line(
            x=piv_2025['Año'], y=piv_2025['Apropiación a precios constantes (2025)'], 
            name='Apropiación a precios constantes (2025)', line=dict(color=DIC_COLORES['ax_viol'][1])
        ),
        row=1, col=1
    )

    piv_tipo_gasto = (data
                      .groupby(['Año', 'Tipo de gasto'])['Apropiación a precios constantes (2025)']
                      .sum()
                      .reset_index())
    piv_tipo_gasto['total'] = piv_tipo_gasto.groupby(['Año'])['Apropiación a precios constantes (2025)'].transform('sum')

    piv_tipo_gasto['%'] = ((piv_tipo_gasto['Apropiación a precios constantes (2025)'] / piv_tipo_gasto['total']) * 100).round(2)

        
    for i, group in piv_tipo_gasto.groupby('Tipo de gasto'):
        fig.add_trace(go.Bar(
            x=group['Año'],
            y=group['%'],
            name=i, marker_color=dict_gasto[i]
        ), row=1, col=2)

    fig.update_layout(barmode='stack', hovermode='x unified')
    fig.update_layout(width=1000, height=500, legend=dict(orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1), title='Histórico general <br><sup>Cifras en miles de millones de pesos</sup>', yaxis_tickformat='.0f')


    st.plotly_chart(fig)





    st.subheader('Sector')

    sectors = data['Sector'].unique()

    sector = st.selectbox("Seleccione el sector", sectors, key=2)

    filter_sector = data[data['Sector'] == sector]

    pivot_sector = filter_sector.pivot_table(index='Año', values='Apropiación a precios constantes (2025)', aggfunc='sum').reset_index()

    fig = make_subplots(rows=1, cols=2, x_title='Año', shared_yaxes=True)
    
    fig.add_trace(
        go.Line(
            x=pivot_sector['Año'], y=pivot_sector['Apropiación a precios constantes (2025)'], 
            name='Apropiación a precios constantes (2025)', line=dict(color=DIC_COLORES['ax_viol'][1])
        ),
        row=1, col=1
    )

    piv_tipo_gasto_sector = (filter_sector
                      .groupby(['Año', 'Tipo de gasto'])['Apropiación a precios constantes (2025)']
                      .sum()
                      .reset_index())
    for i, group in piv_tipo_gasto_sector.groupby('Tipo de gasto'):
        fig.add_trace(go.Bar(
            x=group['Año'],
            y=group['Apropiación a precios constantes (2025)'],
            name=i, marker_color=dict_gasto[i]
        ), row=1, col=2)

    fig.update_layout(barmode='stack', hovermode='x unified')


    fig.update_layout(width=1000, height=500, legend=dict(orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1), title=f"{sector} <br><sup>Cifras en miles de millones de pesos</sup>", yaxis_tickformat='.0f')

    st.plotly_chart(fig)

    st.subheader(f"Variación histórica por sector: {sector}")

    pivot_sector = pivot_sector.set_index('Año')
    pivot_sector['pct'] = pivot_sector['Apropiación a precios constantes (2025)'].pct_change()
    pivot_sector['pct'] = (pivot_sector['pct'] * 100).round(2)
    den = max(pivot_sector.index) - min(pivot_sector.index)
    pivot_sector['CAGR'] = ((pivot_sector.loc[max(pivot_sector.index), 'Apropiación a precios constantes (2025)'] / pivot_sector.loc[min(pivot_sector.index), 'Apropiación a precios constantes (2025)']) ** (1/12)) - 1
    pivot_sector['CAGR'] = (pivot_sector['CAGR'] * 100).round(2)
    pivot_sector = pivot_sector.reset_index()

    fig = make_subplots(rows=1, cols=2, x_title='Año')

    fig.add_trace(
            go.Bar(x=pivot_sector['Año'], y=pivot_sector['Apropiación a precios constantes (2025)'],
                name='Apropiación a precios constantes (2025)', marker_color=DIC_COLORES['ofiscal'][1]),
            row=1, col=1, 
        )

    fig.add_trace(go.Line(
                x=pivot_sector['Año'], 
                y=pivot_sector['pct'], 
                name='Variación porcentual (%)', line=dict(color=DIC_COLORES['ro_am_na'][1])
            ),
            row=1, col=2
        )
    fig.add_trace(
            go.Line(
                x=pivot_sector['Año'], y=pivot_sector['CAGR'], name='Variación anualizada (%)', line=dict(color=DIC_COLORES['verde'][0])
            ),
            row=1, col=2
        )
    fig.update_layout(width=1000, height=500, legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1), hovermode='x unified', yaxis_tickformat='.0f', title=f"{sector} <br><sup>Cifras en miles de millones de pesos</sup>")

    st.plotly_chart(fig)


    st.subheader('Entidad')

 
    entities_sector = filter_sector['Entidad'].unique()
    entidad = st.selectbox("Seleccione la entidad",
                            entities_sector)
    
    filter_entity = filter_sector[filter_sector['Entidad'] == entidad]

    pivot_entity = filter_entity.pivot_table(index='Año',
                                           values='Apropiación a precios constantes (2025)',
                                           aggfunc='sum')
    
    pivot_entity = pivot_entity.reset_index()

    fig = make_subplots(rows=1, cols=2, x_title='Año', shared_yaxes=True)
    
    fig.add_trace(
        go.Line(
            x=pivot_entity['Año'], y=pivot_entity['Apropiación a precios constantes (2025)'], 
            name='Apropiación a precios constantes (2025)', line=dict(color=DIC_COLORES['ax_viol'][1])
        ),
        row=1, col=1
    )
    piv_tipo_gasto_entity = (filter_entity
                      .groupby(['Año', 'Tipo de gasto'])['Apropiación a precios constantes (2025)']
                      .sum()
                      .reset_index())
    for i, group in piv_tipo_gasto_entity.groupby('Tipo de gasto'):
        fig.add_trace(go.Bar(
            x=group['Año'],
            y=group['Apropiación a precios constantes (2025)'],
            name=i, marker_color=dict_gasto[i]
        ), row=1, col=2)

    fig.update_layout(barmode='stack', hovermode='x unified')

    fig.update_layout(width=1000, height=500, legend=dict(orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1), title=f"{entidad} <br><sup>Cifras en miles de millones de pesos</sup>", yaxis_tickformat='.0f')

    st.plotly_chart(fig)

    if pivot_entity['Año'].nunique() <=1:
        st.warning(f"La entidad {entidad} solo tiene información de un año.")
        st.stop()

    st.subheader(f"Variación histórica por entidad: {entidad}")

    pivot_entity = pivot_entity.set_index('Año')
    pivot_entity['pct'] = pivot_entity['Apropiación a precios constantes (2025)'].pct_change()
    pivot_entity['pct'] = (pivot_entity['pct'] * 100).round(2)
    pivot_entity['CAGR'] = ((pivot_entity.loc[max(pivot_entity.index), 'Apropiación a precios constantes (2025)'] / pivot_entity.loc[min(pivot_entity.index), 'Apropiación a precios constantes (2025)'] ) ** (1/12)) - 1
    pivot_entity['CAGR'] = (pivot_entity['CAGR'] * 100).round(2)
    pivot_entity = pivot_entity.reset_index()

    fig = make_subplots(rows=1, cols=2, x_title='Año')

    fig.add_trace(
        go.Bar(x=pivot_entity['Año'], y=pivot_entity['Apropiación a precios constantes (2025)'],
               name='Apropiación a precios constantes (2025)', marker_color=DIC_COLORES['ofiscal'][1]),
        row=1, col=1, 
    )

    fig.add_trace(go.Line(
            x=pivot_entity['Año'], 
            y=pivot_entity['pct'], 
            name='Variación porcentual (%)', line=dict(color=DIC_COLORES['ro_am_na'][1])
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Line(
            x=pivot_entity['Año'], y=pivot_entity['CAGR'], name='Variación anualizada (%)', line=dict(color=DIC_COLORES['verde'][0])
        ),
        row=1, col=2
    )
    fig.update_layout(width=1000, height=500, legend=dict(orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1), hovermode='x unified', yaxis_tickformat='.0f', title=f"{entidad} <br><sup>Cifras en miles de millones de pesos</sup>")

    st.plotly_chart(fig)

    st.subheader("Cambio de gasto por sector (2019 - 2025)")

    data2 = data.pivot_table(index='Sector',
               columns='Año',
               values='Apropiación a precios constantes (2025)',
               aggfunc='sum').sort_values(by=2025).dropna().tail(15).div(1_000_000_000).round(2).reset_index()

    my_range = range(1, len(data2['Sector']) + 1)

    trace1 = go.Scatter(
        x=data2[2025],
        y=list(my_range),
        mode='markers',
        name='2025',
        hovertext=data2['Sector'],
        hoverinfo='text+x',
        marker_color="#2635bf",
        marker_size=10
    )
    trace2 = go.Scatter(
        x=data2[2019],
        y=list(my_range),
        mode='markers',
        name='2019',
        hovertext=data2['Sector'],
        hoverinfo='text+x',
        marker_color="#D8841C",
        marker_size=10
    )

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)


    fig.update_layout(
        title='Cambio del gasto por sector (2019 - 2025)',
        xaxis_title='Gasto (miles de millones de pesos)',
        yaxis_title='',
        yaxis=dict(tickvals=list(my_range), ticktext=["" for i in my_range]),
        xaxis_tickformat="4.",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=1000, height=600)
    for i, j, k in zip(my_range, data2[2019], data2[2025]):
        fig.add_shape(type='line', x0=j, x1=k, y0=i, y1=i, line_color="#2635bf", line_width=0.5) 

    st.plotly_chart(fig)

    st.subheader("Cambio de gasto por entidad (2019 - 2025)")


    data2 = data.pivot_table(index='Entidad',
               columns='Año',
               values='Apropiación a precios constantes (2025)',
               aggfunc='sum').sort_values(by=2025).dropna().tail(15).div(1_000_000_000).round(2).reset_index()

    my_range = range(1, len(data2['Entidad']) + 1)

    trace1 = go.Scatter(
        x=data2[2025],
        y=list(my_range),
        mode='markers',
        name='2025',
        hovertext=data2['Entidad'],
        hoverinfo='text+x',
        marker_color="#2635bf",
        marker_size=10
    )
    
    trace2 = go.Scatter(
        x=data2[2019],
        y=list(my_range),
        mode='markers',
        name='2019',
        hovertext=data2['Entidad'],
        hoverinfo='text+x',
        marker_color="#D8841C",
        marker_size=10
    )

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)


    fig.update_layout(
        title='Cambio del gasto por entidad (2019 - 2025)',
        xaxis_title='Gasto (miles de millones de pesos)',
        yaxis_title='',
        yaxis=dict(tickvals=list(my_range), ticktext=["" for i in my_range]),
        xaxis_tickformat="4.",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=1000, height=600)
    for i, j, k in zip(my_range, data2[2019], data2[2025]):
        fig.add_shape(type='line', x0=j, x1=k, y0=i, y1=i, line_color="#2635bf", line_width=0.5) 
 

    st.plotly_chart(fig)



    








    st.subheader("Descarga de datos")




    binary_output = BytesIO()
    data.to_excel(binary_output, index=False)
    st.download_button(label = 'Descargar datos actualizados',
                    data = binary_output.getvalue(),
                    file_name = 'datos_def_2025.xlsx')

with tab7:

    import json 

    with open("dictios/dic_entities.json") as f:
        dic_entities = json.load(f)

    with open("dictios/dic_sec_ents.json") as f:
        dic_sec_ents = json.load(f)

    with open("dictios/dic_sector.json") as f:
        dic_sector = json.load(f)
    
    df = pd.Series(dic_entities).reset_index()
    df.columns = ['Código de entidad', 'Entidad']
    df['Código de sector'] = df['Código de entidad'].map(dic_sec_ents)
    df['Código de sector'] = df['Código de sector'].astype(str)
    df['Sector'] = df['Código de sector'].map(dic_sector)



    st.subheader('Buscar por entidad:')
    entidad = st.selectbox("Seleccione la entidad: ",  df['Entidad'].unique().tolist())

    filtro_entidad = df[df['Entidad'] == entidad]

    st.write(f"El sector asociado a esta entidad es: {filtro_entidad['Sector'].unique()[0]}")

    st.divider()

    st.subheader("Buscar por sector")

    

    sector = st.selectbox("Seleccione el sector: ", df['Sector'].unique().tolist())
    filtro_sector = df[df['Sector'] == sector]

    st.write(f"Estas son las entidades del sector: {sector}")

    st.dataframe(filtro_sector[['Código de entidad', 'Entidad']])
    st.divider()

    st.subheader("Buscar por coincidencias")

    text = st.text_input("Escriba una palabra o varias que hagan parte del nombre de la entidad. ")
    
    st.dataframe(df[df['Entidad'].str.lower().str.contains(text.lower().strip())])

with tab8:
    st.subheader("Convertir datos csv a excel")

    uploaded_file = st.file_uploader("Cargue un doc CSV:")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        binary_output = BytesIO()
        dataframe.to_excel(binary_output, index=False)
        st.download_button(label = 'Descargar datos convertidos',
                        data = binary_output.getvalue(),
                        file_name = 'datos_convertidos.xlsx')

    










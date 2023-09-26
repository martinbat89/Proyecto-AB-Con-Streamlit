#importamos las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from statsmodels.stats.proportion import proportions_ztest


#Lectura de los datos y carga a la variable data_mk
data_mk1 = pd.read_csv('C:/Users/Martin/Documents/programacion/proyectos/ab_Marketing_20230905/project/marketing_AB.csv')



st.title("Prueba AB Marketing: 📊")
st.markdown('##### En base a un conjunto de datos de precargados conocé como varia una prueba z de proporciones aplicada a una campaña de anuncios.')
st.subheader("Descripción:")
st.write("Para determinar si los anuncios por una campaña tienen efectos en las compras por parte de los clientes se cuenta con un set de datos con el resultado de un proceso experimental. Algunas personas estuvieron expuestas a anuncios (Grupo AD - Experimental) y otras solo recibieron anuncios publicos o ningún anuncio (PSA - De control).")
st.write("A continuación se muestran algunos registros de la tabla analizada para conocer los campos disponibles:")


# Creamos sidebar:
side_tit =  st.sidebar.header('Seleccioná las opciones')

#sidebar
max_filas = data_mk1.shape[0]
side_muestra = st.sidebar.slider('Cantidad De Datos', min_value=100, max_value=max_filas)
data_mk = data_mk1.sample(n=side_muestra, replace=True)
st.dataframe(data_mk.head(2))
st.caption("Para facilitar el análisis se decidió utilizar solo los campos que indican la exposición a annuncios (Test Group) y si la persona compro (Converted).")
         
conf_radio = st.sidebar.radio('Nivel de Confianza:', ["99%", "95%", "90%"])

if conf_radio == "99%":
    alpha = 0.01
elif conf_radio == "95%":
    alpha = 0.05
else:
    alpha = 0.1

st.sidebar.metric('Alpha', alpha)
st.sidebar.caption("Dataset descargado en Kaggle. Cuenta con Licencia CCO: Public Domain.")
st.sidebar.write("[Consulta el Dataset en Kaggle](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing?resource=download)")

st.subheader("¿Porque un z test de proporciones?")
st.write("Luego de agrupar los datos se determino que para comparar los resultados de la campaña, resulta necesario analizar que proporcion de personas que recibieron anuncios generaron compras y que proporcion de los que no recibieron anuncios, realizaron compras. De esta forma verificar si la campaña genera efectos en la conversión. A continuación se muestra visualmente los resultados del agrupamiento de datos realizado en el análisis exploratorio de datos:")


#GRAFICOS
#agrupar por tipo de cliente
tipo_adv = data_mk.groupby(by=["test group"])["user id"].count()

arrData45 = np.array(tipo_adv)
arrLab = tipo_adv.keys()

fig, ax = plt.subplots()
ax.pie(arrData45, labels=arrLab, autopct='%1.1f%%', colors=["grey", "blue"])


#agrupamos los registros por la col "converted" y "test group"
totConv = data_mk.groupby(by=["converted", "test group"]).count()["user id"]
keysVar = totConv.keys()
arrLab_com = []
arrData_comb = []

for llave in keysVar:
    if llave[0] == False:
        conv = "No Compro"
    else:
        conv = "Compro"
        
    arrLab_com.append(conv + "_" + llave[1])

for dato in totConv:
    arrData_comb.append(dato)

fig2, ax2 = plt.subplots()
ax2.pie(arrData_comb, labels=arrLab_com, autopct='%1.1f%%', colors=["grey", "blue"])


col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig, use_container_width=True)
with col2:
    st.info("En el gráfico de la izquierda se muestra el porcentaje de personas que recibieron anuncios (AD) y el que no (PSA). Se observa que el grupo de mayor relevancia es el experimental (AD).")
st.write(" ")
col1, col2 = st.columns(2)
with col1:
    st.info("En el gráfico derecho se muestra el porcentaje de conversion para cada grupo del experimento (AD con compra, AD sin compra, PSA con compra, PSA sin compra).")

with col2:
    
    st.pyplot(fig2, use_container_width=True)

##seccion de conversiones

#calculos
#Quitar espacio de nombres de campos. 
data_mk.rename(columns=lambda e: e.strip().replace(" ", "_"), inplace=True)

# convertimos columna de conversión a valores enteros

data_mk['converted'] = data_mk['converted'].astype(int)

#Filtramos el df y en uno dejamos solo los que tuvieron aviso y en otro los que no
tratamiento_ad = data_mk.query('test_group == "ad"')
control_psa = data_mk.query('test_group == "psa"')


#Cálculamos los valores de conversion de los tres df
conversion_adv = str(round(tratamiento_ad['converted'].mean() * 100,0)) + " %"
conversion_psa = str(round(control_psa['converted'].mean() * 100,0)) + " %"
conversion_global = str(round(data_mk['converted'].mean() * 100,0)) + " %"



st.write(" ")
st.markdown('##### Tasas de Conversión')
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Conversión AD', conversion_adv)

with col2:
    st.metric('Conversión PSA', conversion_psa)

with col3:
    st.metric('Conversión Global', conversion_global)

st.write("Analizando los valores obtenidos se observa que la proporcion de conversion de los que recibieron anuncios es mayor. Sin embargo...")
st.warning("¿Es esta mayor proporción estadisticamente representativa? Para determinarlo es necesario realizar una prueba de hipotesis de proporciones.")

#Prueba z Test
st.subheader("Definición de hipótesis H0 y H1")
st.write("* _H0: No existen diferencias de conversion entre hacer o no hacer campaña publicitaria_")
st.write("* H0: Si existen diferencias de conversion entre hacer o no hacer campaña publicitaria")

st.subheader("Definición Estadístico de Contraste (pValor)")

#st.caption('Balloons. Hundreds of them...')

##Ztest Final
#creamos variables y las cargamos a arrays de numpy porque asi lo pide la fc
conv_adv = len(tratamiento_ad.query('converted == 1'))
tot_adv = len(tratamiento_ad)
conv_psa = len(control_psa.query('converted == 1')) 
tot_psa = len(control_psa) 

#creamos listas numpy
arr_convert = np.array([conv_adv, conv_psa])
arr_vis = np.array([tot_adv, tot_psa])


#calculos con ZTest en ambas direcciones
pvalor = proportions_ztest(count=arr_convert, nobs=arr_vis)[1]
tvalor = proportions_ztest(count=arr_convert, nobs=arr_vis)[0]
st.caption("Se deja el parámetro 'Alternative' de la funcion utilizada que plantea por defecto probar si hay una diferencia significativa en ambas direcciones.")
st.write("Si el Pvalor es menor a Alpha rechazamos HO. Caso contrariio no rechazamos H0.")
st.dataframe({
    "pvalor": pvalor,
    "tvalor": tvalor
})

st.subheader("Resultado Z Test")

if pvalor<alpha:
    st.success("Rechazo H0: Se confirma que los que vieron el anuncio compraron mas")
else:
    st.warning("No Rechazo H0: No existe evidencia estadística que nos confirme que ver el anuncio impacte en la compra")

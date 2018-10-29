# English
Description of the project in English.


# Español
Este proyecto tiene como objetivo modelar textos anotados con informacion espacio-temporal (tweets o los relatos geoetiquetados), usualmente tuplas en la forma <texto, lugar, tiempo>. El <lugar> es usualmente un par <latitud,longitud> y el <tiempo> un timestamp. El objetivo de los modelos es ser capaces de responder consultas de alguno de los tres elementos con los otros. Por ejemplo: dado un lugar y horario cual es el texto que caracteriza ese lugar y horario. Dada una palabra cual es horario o el lugar o horario que mejor se asocia a esa palabra.
El procesamiento de las tuplas es a traves de un fichero que representa un dataframe de la biblioteca pandas de python.
El fichero debe estar en formato .pickle o .csv. Este codigo no se encarga de descargar tweets o algun otro tipo de datos geoetiquetados, el codigo asume que ya se poseen esos datos. El fichero debe tener cuatro columnas nombradas 'created_at', 'latitude', 'longitude' y 'texts'.
El fichero utils.py esta el codigo correspondiente a la carga y preprocesamiento de los datos. En el preprocesamiento se convierten todas las palabras a minusculas, se eliminan stop-words y se eliminan palabras no alfanumericas.Tambien se discretizan las coordenadas y el timestamp de acuerdo a la granularidad deseada.
El jupyter notebook Represent_As_Doc.ipynb tiene los modelos de representacion como textos. Estos modelos consisten en: representar cada ventana temporal como la agregacion de los textos que aparecen en la ventana, representar cada celda  espacial como la agregacion de los textos que aparecen en la ventana y representar cada palabra como la agregacion de los textos en que aparece esa palabra. Luego se utilizan modelos de representacion de textos como TF, TF-IDF, LDA. En los experimentos realizados los mejores resultados se obtuvieron con TF. El codigo presentado en Represent_As_Doc.ipynb esta diseñado para trabajar en google-colaboratory aunque es facilmente adaptable a ambientes locales.
La construccion de los modelos toma un tiempo en el rango de los 5 minutos.
 


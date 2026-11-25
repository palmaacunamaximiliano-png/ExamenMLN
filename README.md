# Aeropuertos críticos y recomendación de rutas en la red mundial de vuelos

**Integrantes:** Maximiliano Palma y Joaquín Hidalgo · Examen electivo "Aprendizaje no supervisado – Análisis de redes"

## Pregunta del proyecto

¿Qué aeropuertos son críticos para mantener conectada la red mundial de rutas aéreas y qué rutas nuevas convendría sugerir para reducir esa dependencia?

## Fuente de la red

[OpenFlights](https://openflights.org/data.html) (`airports.dat` y `routes.dat`, licencia ODbL; datos de aproximadamente 2014).
Nodo = aeropuerto. Arista = existe al menos una ruta comercial directa entre dos aeropuertos (sin dirección ni pesos).
3.214 aeropuertos y 18.858 rutas.

## App publicada

> Pegar aquí el link de Streamlit Community Cloud una vez publicada.

## Qué contiene el dashboard

| Pestaña | Contenido |
|---|---|
| Resumen de la red | Nodos, aristas, densidad, diámetro, modularidad y comparación con una red aleatoria |
| Explorador de actores | Mapa con color por comunidad, rol o país y tamaño por centralidad; red del ego; tabla de centralidades con descarga |
| Roles y comunidades | Mapa del ACP coloreado por rol, centroides, tabla de contingencia rol × comunidad |
| Recomendador | Recomendaciones para un aeropuerto o para un aeropuerto hipotético, con motivo, puntaje, comunidad de destino y marca de respaldo de populares |
| Evaluación | HitRate@N y AUC de cada método frente a la línea base |
| Simulador what-if | Fragmentación de la red por estrategia y porcentaje de aeropuertos removidos |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
├── README.md
├── requirements.txt
├── app.py
├── data/            nodos.csv, aristas.csv, recomendaciones.csv, hitrate.csv, whatif.csv y tablas de resumen
└── Examen_Grupo_X.ipynb   (generado por el notebook: sección 10)
```

Los archivos de `data/` los genera la sección 10 del notebook; la app solo los lee.

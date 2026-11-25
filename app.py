"""Dashboard: criticidad y recomendación de rutas en la red mundial de aeropuertos (OpenFlights).
Lee los archivos precalculados de data/ que genera la sección 10 del notebook."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Red mundial de rutas aéreas", page_icon="✈️", layout="wide")

METODOS = {"PageRank personalizado": "ppr", "Adamic–Adar": "adamic_adar", "Jaccard": "jaccard",
           "Vecinos comunes": "vecinos_comunes", "Más populares (línea base)": "populares"}
MODOS = {"Sin ajuste de comunidad": "ninguno",
         "Dentro de la comunidad (cierre triádico)": "dentro",
         "Puentes hacia otras comunidades": "puentes"}
ROLES = ["Hub", "Integrado de comunidad", "Periferia abierta", "Periferia cerrada", "Periferia lejana", "Terminal"]
CENTRALIDADES = ["degree", "closeness", "betweenness", "eigenvector", "pagerank", "clustering"]


# ------------------------------------------------------------------ datos
@st.cache_data
def cargar():
    d = {"nodos": pd.read_csv(DATA / "nodos.csv", index_col="id"), "aristas": pd.read_csv(DATA / "aristas.csv"),
         "hitrate": pd.read_csv(DATA / "hitrate.csv"), "whatif": pd.read_csv(DATA / "whatif.csv"),
         "whatif_rutas": pd.read_csv(DATA / "whatif_rutas.csv"), "curva_K": pd.read_csv(DATA / "curva_K.csv"),
         "comparacion": pd.read_csv(DATA / "comparacion.csv", index_col="medida"),
         "comunidades": pd.read_csv(DATA / "comunidades.csv", index_col="medida"),
         "centroides": pd.read_csv(DATA / "centroides_roles.csv"),
         "varianza": pd.read_csv(DATA / "varianza_pca.csv", index_col="componente")["varianza"],
         "roles": pd.read_csv(DATA / "roles.csv", index_col="rol")}
    return d


@st.cache_resource
def red_base():
    """Matriz de adyacencia (sin pesos) en el mismo orden que nodos.csv."""
    d = cargar()
    idx = {n: i for i, n in enumerate(d["nodos"].index)}
    f, c = d["aristas"]["origen"].map(idx).to_numpy(), d["aristas"]["destino"].map(idx).to_numpy()
    n = len(idx)
    M = sparse.coo_matrix((np.ones(len(f)), (f, c)), shape=(n, n))
    return ((M + M.T) > 0).astype(float).tocsr()


@st.cache_resource
def grafo_nx():
    d = cargar()
    return nx.from_pandas_edgelist(d["aristas"], "origen", "destino")


def preparar(M):
    grado = np.asarray(M.sum(axis=1)).ravel()
    inv_log = np.where(grado > 1, 1.0 / np.log(np.maximum(grado, 2)), 0.0)
    P = (sparse.diags(1.0 / np.maximum(grado, 1)) @ M).tocsr()
    return {"M": M, "grado": grado, "inv_log": inv_log, "P": P}


# ------------------------------------------------------------------ puntajes (los mismos de la sección 6 del notebook)
def puntajes(E, i, metodo, alfa=0.85, iteraciones=50):
    M, grado = E["M"], E["grado"]
    fila = M[[i], :]
    if metodo == "populares":
        return grado.copy()
    if metodo == "vecinos_comunes":
        return np.asarray((fila @ M).todense()).ravel()
    if metodo == "jaccard":
        cn = np.asarray((fila @ M).todense()).ravel()
        union = grado[i] + grado - cn
        return np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    if metodo == "adamic_adar":
        return np.asarray(((fila.multiply(E["inv_log"])) @ M).todense()).ravel()
    if metodo == "ppr":
        e = np.zeros(len(grado)); e[i] = 1.0
        r = e.copy()
        for _ in range(iteraciones):
            r = (1 - alfa) * e + alfa * (E["P"].T @ r)
        return r
    raise ValueError(metodo)


def recomendar(E, i, info, com_arr, top, metodo, modo, lam):
    """Igual que `recomendar` del notebook, pero sobre índices. info: DataFrame con etiqueta, ciudad, pais."""
    indptr, indices = E["M"].indptr, E["M"].indices
    vecinos_i = indices[indptr[i]:indptr[i + 1]]
    base = puntajes(E, i, metodo).astype(float)
    s = base.copy()
    if modo != "ninguno" and lam > 0 and len(vecinos_i) > 0:
        mismo = com_arr == com_arr[i]
        s = s * (1 + lam * (mismo if modo == "dentro" else ~mismo))
    excluidos = np.zeros(len(s), dtype=bool)
    excluidos[i] = True
    excluidos[vecinos_i] = True
    s[excluidos] = -np.inf
    del_metodo = [j for j in np.argsort(-s, kind="stable")[:top] if s[j] > 0] if len(vecinos_i) > 0 else []
    respaldo = []
    if len(del_metodo) < top:
        for j in np.argsort(-E["grado"], kind="stable"):
            if not excluidos[j] and j not in del_metodo:
                respaldo.append(j)
            if len(del_metodo) + len(respaldo) == top:
                break
    filas = []
    for j, es_respaldo in [(j, False) for j in del_metodo] + [(j, True) for j in respaldo]:
        comunes = np.intersect1d(vecinos_i, indices[indptr[j]:indptr[j + 1]])
        otra = com_arr[j] != com_arr[i]
        if es_respaldo:
            motivo = f"respaldo de populares ({int(E['grado'][j])} rutas)"
        elif len(comunes) > 0:
            nombres = [info.iloc[c]["etiqueta"] for c in comunes[np.argsort(-E["grado"][comunes])][:3]]
            motivo = f"{len(comunes)} vecino{'s' if len(comunes) > 1 else ''} en común ({', '.join(nombres)})"
        else:
            motivo = "sin vecinos en común (llega por paseo aleatorio)"
        filas.append({"j": j, "Aeropuerto": info.iloc[j]["etiqueta"], "Ciudad": info.iloc[j]["ciudad"],
                      "País": info.iloc[j]["pais"], "Puntaje": float(E["grado"][j] if es_respaldo else base[j]),
                      "Comunidad destino": int(com_arr[j]), "¿Otra comunidad?": "sí" if otra else "no",
                      "Vecinos en común": len(comunes), "Respaldo de populares": "sí" if es_respaldo else "no",
                      "Motivo": motivo})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ carga
D = cargar()
nodos = D["nodos"]
M0 = red_base()
E0 = preparar(M0)
info0 = nodos[["etiqueta", "ciudad", "pais"]]
com0 = nodos["comunidad"].to_numpy()
grado0 = E0["grado"]
opciones = (nodos["etiqueta"] + " · " + nodos["ciudad"].fillna("") + " · " + nodos["pais"].fillna("")
            + " (" + pd.Series(grado0.astype(int), index=nodos.index).astype(str) + " rutas)")
etiqueta_a_idx = {t: i for i, t in enumerate(opciones)}

st.title("✈️ Red mundial de rutas aéreas: aeropuertos críticos y recomendación de rutas")
st.caption("Pregunta del proyecto: ¿qué aeropuertos son críticos para mantener conectada la red mundial de rutas aéreas "
           "y qué rutas nuevas convendría sugerir para reducir esa dependencia? Fuente: OpenFlights (datos de 2014).")

t_res, t_exp, t_rol, t_rec, t_eval, t_wi = st.tabs(
    ["Resumen de la red", "Explorador de actores", "Roles y comunidades", "Recomendador", "Evaluación", "Simulador what-if"])

# ================================================================== 1. RESUMEN
with t_res:
    comp, com_res = D["comparacion"], D["comunidades"]
    c = st.columns(5)
    c[0].metric("Aeropuertos (nodos)", f"{int(comp.loc['Nodos', 'OpenFlights']):,}".replace(",", "."))
    c[1].metric("Rutas (aristas)", f"{int(comp.loc['Aristas', 'OpenFlights']):,}".replace(",", "."))
    c[2].metric("Densidad", f"{comp.loc['Densidad', 'OpenFlights']:.4f}")
    c[3].metric("Diámetro", int(comp.loc["Diámetro", "OpenFlights"]), help="Medido en la componente gigante.")
    c[4].metric("Modularidad (Louvain)", f"{com_res.loc['modularidad', 'Louvain']:.3f}")
    st.subheader("OpenFlights frente a una red aleatoria con los mismos nodos y aristas")
    tabla = comp.copy()
    tabla["OpenFlights / aleatoria"] = (tabla["OpenFlights"] / tabla["Erdős–Rényi"]).round(2)
    st.dataframe(tabla.style.format({"OpenFlights": "{:,.4f}", "Erdős–Rényi": "{:,.4f}", "OpenFlights / aleatoria": "{:,.2f}"}),
                 width="stretch")
    st.markdown(
        f"La red real tiene un **clustering {comp.loc['Clustering medio', 'OpenFlights'] / comp.loc['Clustering medio', 'Erdős–Rényi']:.0f} veces mayor** "
        f"y un **grado máximo {comp.loc['Grado máximo', 'OpenFlights'] / comp.loc['Grado máximo', 'Erdős–Rényi']:.1f} veces mayor** que una red al azar: "
        "unos pocos hubs concentran las rutas y los vecinos de un aeropuerto tienden a estar conectados entre sí. "
        "Los caminos son cortos (unos 4 vuelos de media) pese a la baja densidad.")
    st.subheader("Comunidades")
    st.dataframe(com_res.style.format("{:,.4f}"), width="stretch")
    st.caption("Louvain detecta menos comunidades y con mayor modularidad que greedy; se usa Louvain (semilla fija) en todo el análisis.")

# ================================================================== 2. EXPLORADOR
with t_exp:
    c1, c2, c3 = st.columns(3)
    color_por = c1.selectbox("Colorear por", ["Comunidad", "Rol estructural", "País"], key="color")
    tam_por = c2.selectbox("Tamaño según centralidad", CENTRALIDADES, index=2, key="tam")
    solo_gig = c3.checkbox("Solo componente gigante", value=False)
    df = nodos.copy()
    if solo_gig:
        df = df[df["rol"].notna()]
    if color_por == "Comunidad":
        df["color"] = df["comunidad"].astype(str)
    elif color_por == "Rol estructural":
        df["color"] = df["rol"].fillna("fuera de la componente gigante")
    else:
        top_p = df["pais"].value_counts().head(12).index
        df["color"] = df["pais"].where(df["pais"].isin(top_p), "otros")
    v = df[tam_por].astype(float)
    df["tam"] = 3 + 22 * np.sqrt(v / v.max())
    fig = px.scatter_geo(df, lat="lat", lon="lon", color="color", size="tam", size_max=22, hover_name="etiqueta",
                         hover_data={"ciudad": True, "pais": True, "degree": True, "comunidad": True, "rol": True,
                                     "tam": False, "color": False, "lat": False, "lon": False},
                         projection="natural earth", opacity=0.75)
    fig.update_geos(showcountries=True, countrycolor="#cccccc", landcolor="#f4f4f4", oceancolor="#eaf2f8", showocean=True)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), legend_title_text=color_por)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Red del ego de un aeropuerto")
    sel = st.selectbox("Busca un aeropuerto por código, ciudad o país", list(opciones),
                       index=list(opciones).index(next(t for t in opciones if t.startswith("SCL ·"))), key="ego")
    ego_id = nodos.index[etiqueta_a_idx[sel]]
    Gx = grafo_nx()
    vec = list(Gx[ego_id])
    sub = Gx.subgraph([ego_id] + vec)
    pos = nx.spring_layout(sub, seed=42, k=0.6 / np.sqrt(max(len(sub), 1)))
    ex, ey = [], []
    for a, b in sub.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    nodos_sub = list(sub.nodes)
    figg = go.Figure()
    figg.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="#bbbbbb"), hoverinfo="skip", showlegend=False))
    figg.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodos_sub], y=[pos[n][1] for n in nodos_sub], mode="markers+text",
        text=[nodos.loc[n, "etiqueta"] if (n == ego_id or Gx.degree(n) >= 60) else "" for n in nodos_sub],
        textposition="top center", hovertext=[f"{nodos.loc[n, 'etiqueta']} · {nodos.loc[n, 'ciudad']} ({nodos.loc[n, 'pais']})" for n in nodos_sub],
        marker=dict(size=[16 if n == ego_id else 5 + 10 * np.sqrt(Gx.degree(n) / 248) for n in nodos_sub],
                    color=[nodos.loc[n, "comunidad"] for n in nodos_sub], colorscale="Turbo", line=dict(width=0.5, color="white")),
        showlegend=False))
    figg.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                       title=f"{nodos.loc[ego_id, 'etiqueta']} y sus {len(vec)} vecinos (color = comunidad)")
    st.plotly_chart(figg, width="stretch")

    st.subheader("Tabla de centralidades")
    tabla = nodos[["etiqueta", "ciudad", "pais", "comunidad", "rol"] + CENTRALIDADES].rename(columns={"etiqueta": "aeropuerto"})
    texto = st.text_input("Filtrar por código, ciudad o país", "")
    if texto:
        m = tabla.apply(lambda col: col.astype(str).str.contains(texto, case=False, na=False)).any(axis=1)
        tabla = tabla[m]
    st.dataframe(tabla, width="stretch", height=380)
    st.download_button("Descargar tabla (CSV)", tabla.to_csv().encode("utf-8"), file_name="centralidades.csv", mime="text/csv")

# ================================================================== 3. ROLES Y COMUNIDADES
with t_rol:
    gig = nodos[nodos["rol"].notna()]
    var = D["varianza"]
    figr = px.scatter(gig, x="PC1", y="PC2", color="rol", hover_name="etiqueta", hover_data=["ciudad", "pais", "degree"],
                      category_orders={"rol": ROLES}, opacity=0.7)
    cen = D["centroides"]
    figr.add_trace(go.Scatter(x=cen["PC1"], y=cen["PC2"], mode="markers+text", text=cen["rol"], textposition="top center",
                              marker=dict(symbol="x", size=12, color="black"), name="Centroides", hoverinfo="skip"))
    figr.update_layout(height=560, xaxis_title=f"PC1 ({var['PC1']:.0%}): popularidad y cercanía",
                       yaxis_title=f"PC2 ({var['PC2']:.0%}): cohesión (+) frente a intermediación (−)",
                       title="Roles estructurales en el mapa del ACP (componente gigante)")
    st.plotly_chart(figr, width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Perfil de cada rol (medianas)")
        st.dataframe(D["roles"].round(3), width="stretch")
    with c2:
        st.subheader("Acuerdo entre roles y comunidades")
        st.metric("ARI", f"{adjusted_rand_score(gig['rol'], gig['comunidad']):.3f}")
        st.metric("NMI", f"{normalized_mutual_info_score(gig['rol'], gig['comunidad']):.3f}")
        st.caption("Valores cercanos a 0: los roles son casi independientes de las comunidades.")
    st.subheader("Tabla de contingencia: rol × comunidad (5 comunidades mayores y el resto)")
    grupo = gig["comunidad"].map(lambda c: str(int(c)) if c < 5 else "otras")
    cont = pd.crosstab(gig["rol"], grupo).reindex(ROLES)[["0", "1", "2", "3", "4", "otras"]]
    st.dataframe(cont, width="stretch")

# ================================================================== 4. RECOMENDADOR
with t_rec:
    c1, c2, c3, c4 = st.columns(4)
    metodo_txt = c1.selectbox("Método", list(METODOS), index=0,
                              help="PageRank personalizado fue el mejor método en la evaluación offline.")
    modo_txt = c2.selectbox("Variante de comunidad", list(MODOS), index=0)
    modo = MODOS[modo_txt]
    lam = c3.slider("Fuerza de la variante (λ)", 0.0, 2.0, 0.5, 0.25, disabled=(modo == "ninguno"),
                    help="Multiplica el puntaje por (1 + λ) para los destinos que cumplen la condición de comunidad.")
    top = c4.slider("Recomendaciones", 3, 20, 10)

    modo_uso = st.radio("¿Qué quieres recomendar?", ["Un aeropuerto de la red", "Un conjunto de vecinos que armo yo"], horizontal=True)
    if modo_uso == "Un aeropuerto de la red":
        elegido = st.selectbox("Aeropuerto (escribe para buscar por código, ciudad o país)", list(opciones),
                               index=list(opciones).index(next(t for t in opciones if t.startswith("SCL ·"))), key="rec_aero")
        i = etiqueta_a_idx[elegido]
        frio = st.checkbox("Simular arranque en frío: quitarle todas sus rutas", value=False,
                           help="Muestra qué recomienda el sistema para un aeropuerto sin información.")
        if frio:
            mascara = np.ones(len(nodos)); mascara[i] = 0
            Dg = sparse.diags(mascara)
            E = preparar((Dg @ M0 @ Dg).tocsr())
        else:
            E = E0
        info, com = info0, com0
        origen_txt = nodos.iloc[i]["etiqueta"]
        origen_lat, origen_lon = nodos.iloc[i]["lat"], nodos.iloc[i]["lon"]
        n_rutas = int(E["grado"][i])
        rol = nodos.iloc[i]["rol"] if pd.notna(nodos.iloc[i]["rol"]) else "fuera de la componente gigante"
        k = st.columns(4)
        k[0].metric("Aeropuerto", origen_txt)
        k[1].metric("Rutas actuales", n_rutas)
        k[2].metric("Comunidad", int(com[i]))
        k[3].metric("Rol estructural", rol)
    else:
        elegidos = st.multiselect("Aeropuertos a los que se conectaría el aeropuerto nuevo (mínimo 1)", list(opciones),
                                  default=[t for t in opciones if t.startswith(("SCL ·", "LIM ·", "EZE ·"))][:3])
        if not elegidos:
            st.info("Elige al menos un aeropuerto para armar el conjunto de vecinos.")
            st.stop()
        vecinos = [etiqueta_a_idx[t] for t in elegidos]
        n = len(nodos)
        fila = sparse.csr_matrix((np.ones(len(vecinos)), ([0] * len(vecinos), vecinos)), shape=(1, n))
        E = preparar(sparse.bmat([[M0, fila.T], [fila, None]], format="csr"))      # aeropuerto nuevo unido a los elegidos
        i = n
        com_virtual = int(pd.Series(com0[vecinos]).mode().iloc[0])
        com = np.append(com0, com_virtual)
        info = pd.concat([info0, pd.DataFrame([{"etiqueta": "NUEVO", "ciudad": "aeropuerto hipotético", "pais": ""}])], ignore_index=True)
        origen_txt = "NUEVO"
        origen_lat, origen_lon = nodos.iloc[vecinos]["lat"].mean(), nodos.iloc[vecinos]["lon"].mean()
        n_rutas = len(vecinos)
        k = st.columns(2)
        k[0].metric("Rutas del aeropuerto hipotético", n_rutas)
        k[1].metric("Comunidad asignada (la más frecuente entre sus vecinos)", com_virtual)

    res = recomendar(E, i, info, com, top, METODOS[metodo_txt], modo, lam)
    n_resp = int((res["Respaldo de populares"] == "sí").sum())
    if n_rutas == 0:
        st.warning(f"Arranque en frío: el aeropuerto no tiene rutas, así que las {top} recomendaciones salen de los más populares.")
    elif n_resp > 0:
        st.warning(f"Se usó el respaldo de populares en {n_resp} de {top} recomendaciones (el método no tuvo suficientes candidatos).")
    else:
        st.success(f"Las {top} recomendaciones salen del método «{metodo_txt}».")

    tabla = res.drop(columns="j").copy()
    tabla.insert(0, "N°", range(1, len(tabla) + 1))
    tabla["Puntaje"] = tabla["Puntaje"].map(lambda x: f"{x:.4f}" if x < 10 else f"{x:.0f}")
    st.dataframe(tabla.set_index("N°"), width="stretch")
    st.download_button("Descargar recomendaciones (CSV)", res.drop(columns="j").to_csv(index=False).encode("utf-8"),
                       file_name=f"recomendaciones_{origen_txt}.csv", mime="text/csv")
    k = st.columns(2)
    k[0].metric("Recomendaciones hacia otra comunidad", f"{(res['¿Otra comunidad?'] == 'sí').mean():.0%}")
    k[1].metric("Vecinos en común (promedio)", f"{res['Vecinos en común'].mean():.1f}")

    figm = go.Figure()
    for _, r in res.iterrows():
        j = int(r["j"])
        figm.add_trace(go.Scattergeo(lat=[origen_lat, nodos.iloc[j]["lat"]], lon=[origen_lon, nodos.iloc[j]["lon"]], mode="lines",
                                     line=dict(width=1.5, color="#c0392b" if r["Respaldo de populares"] == "sí" else "#2471a3"),
                                     hoverinfo="skip", showlegend=False))
    figm.add_trace(go.Scattergeo(lat=nodos.iloc[res["j"]]["lat"], lon=nodos.iloc[res["j"]]["lon"], mode="markers+text",
                                 text=res["Aeropuerto"], textposition="top center", marker=dict(size=8, color="#2471a3"),
                                 name="Destinos sugeridos", hovertext=res["Ciudad"] + " (" + res["Motivo"] + ")"))
    figm.add_trace(go.Scattergeo(lat=[origen_lat], lon=[origen_lon], mode="markers+text", text=[origen_txt],
                                 textposition="bottom center", marker=dict(size=12, color="#1e8449", symbol="star"), name="Origen"))
    figm.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#bbbbbb", showland=True,
                     landcolor="#f2f2f2", oceancolor="#e8f1f8", showocean=True)
    figm.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(figm, width="stretch")
    st.caption("Líneas azules: recomendaciones del método. Líneas rojas: respaldo de populares.")
    with st.expander("¿Cómo se calcula?"):
        st.markdown(
            "- **Vecinos comunes**: cuántos destinos comparten dos aeropuertos.\n"
            "- **Jaccard**: lo mismo, corregido por tamaño.\n"
            "- **Adamic–Adar**: pesa más los vecinos poco conectados.\n"
            "- **PageRank personalizado**: paseo aleatorio con reinicio desde el aeropuerto.\n"
            "- **Respaldo de populares**: si el aeropuerto no tiene rutas (arranque en frío) o el método no genera "
            "suficientes candidatos, se completa con los aeropuertos de mayor grado.\n"
            "- **Dentro / puentes**: premian destinos de la misma comunidad (cierre triádico) o de otra comunidad.")

# ================================================================== 5. EVALUACIÓN
with t_eval:
    h = D["hitrate"].set_index("variante")
    st.markdown("Prueba de ocultamiento: se esconde el 20% de las rutas y se mide si el recomendador las recupera. "
                "Promedio de 5 particiones aleatorias (la desviación estándar entre particiones es de 0,016 o menos, salvo en arranque en frío).")
    base = ["azar", "populares", "jaccard", "vecinos_comunes", "adamic_adar", "ppr"]
    Ns = [1, 3, 5, 10]
    figh = go.Figure()
    for v in base:
        figh.add_trace(go.Scatter(x=Ns, y=[h.loc[v, f"HitRate@{k}"] for k in Ns], mode="lines+markers", name=v,
                                  error_y=dict(type="data", array=[h.loc[v, f"HitRate@{k}_desv"] for k in Ns], visible=True)))
    figh.update_layout(height=420, xaxis=dict(title="N (recomendaciones por aeropuerto)", tickvals=Ns), yaxis_title="HitRate@N",
                       title="HitRate@N: fracción de rutas ocultas recuperadas")
    c1, c2 = st.columns(2)
    c1.plotly_chart(figh, width="stretch")
    figa = px.bar(h.loc[base].reset_index(), x="variante", y="AUC", error_y="AUC_desv", range_y=[0.4, 1.0],
                  title="AUC (0,5 = azar, 1 = perfecto)")
    figa.update_layout(height=420, xaxis_title="")
    c2.plotly_chart(figa, width="stretch")
    st.subheader("Tabla completa")
    cols = ["HitRate@1", "HitRate@3", "HitRate@5", "HitRate@10", "HitRate@10 cruzadas", "HitRate@10 grado<=1", "AUC", "cobertura@10", "novedad@10"]
    st.dataframe(h[cols].round(3), width="stretch")
    st.download_button("Descargar evaluación (CSV)", D["hitrate"].to_csv(index=False).encode("utf-8"), file_name="hitrate.csv", mime="text/csv")
    st.subheader("Precisión frente a novedad")
    orden_var = ["ppr + dentro (λ=1)", "ppr + dentro (λ=0.5)", "ppr", "ppr + puentes (λ=0.25)", "ppr + puentes (λ=0.5)", "ppr + puentes (λ=1)"]
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=h.loc[orden_var, "novedad@10"], y=h.loc[orden_var, "HitRate@10"], mode="lines+markers",
                              name="HitRate@10 (todas las rutas)", text=orden_var))
    figp.add_trace(go.Scatter(x=h.loc[orden_var, "novedad@10"], y=h.loc[orden_var, "HitRate@10 cruzadas"], mode="lines+markers",
                              name="HitRate@10 (rutas entre comunidades)", text=orden_var))
    figp.update_layout(height=400, xaxis_title="Novedad@10 (% de recomendaciones hacia otra comunidad)", yaxis_title="HitRate@10")
    st.plotly_chart(figp, width="stretch")
    st.caption("Un acierto es recuperar una ruta que ya existía en 2014; no mide demanda ni rentabilidad.")

# ================================================================== 6. WHAT-IF
with t_wi:
    w = D["whatif"]
    estrategias = list(w["estrategia"].unique())
    c1, c2, c3 = st.columns([2, 1, 1])
    elegidas = c1.multiselect("Estrategias de remoción", estrategias, default=estrategias)
    pct = c2.slider("% de aeropuertos removidos", 1, 10, 5)
    metrica = c3.selectbox("Métrica", ["gigante", "componentes", "camino", "eficiencia"],
                           format_func=lambda m: {"gigante": "Componente gigante", "componentes": "Número de componentes",
                                                  "camino": "Longitud de camino", "eficiencia": "Eficiencia global"}[m])
    figw = go.Figure()
    for e in elegidas:
        s = w[w["estrategia"] == e].sort_values("pct")
        figw.add_trace(go.Scatter(x=s["pct"], y=s[f"{metrica}_mean"], mode="lines+markers", name=e,
                                  line=dict(dash="dash") if e == "azar" else None,
                                  error_y=dict(type="data", array=s[f"{metrica}_std"], visible=e in ("azar", "rol Hub (azar dentro del rol)"))))
    figw.add_vline(x=pct, line_dash="dot", line_color="gray")
    figw.update_layout(height=450, xaxis_title="% de aeropuertos removidos", yaxis_title=metrica)
    st.plotly_chart(figw, width="stretch")
    n_rem = round(pct * 3188 / 100)
    st.markdown(f"**Con {pct}% removido ({n_rem} aeropuertos de la componente gigante):**")
    aqui = w[(w["pct"] == pct) & (w["estrategia"].isin(elegidas))].set_index("estrategia")
    st.dataframe(aqui[["gigante_mean", "componentes_mean", "camino_mean", "eficiencia_mean"]]
                 .rename(columns={"gigante_mean": "componente gigante", "componentes_mean": "componentes",
                                  "camino_mean": "camino medio", "eficiencia_mean": "eficiencia"}).round(3), width="stretch")
    st.caption("degree, betweenness y pagerank: orden calculado una vez sobre la red intacta. Azar y rol Hub: promedio de 20 repeticiones. "
               "Longitud de camino y eficiencia se estiman con 500 aeropuertos de origen.")

    st.subheader("¿Qué rutas nuevas reducen la dependencia?")
    st.markdown("Se da una segunda ruta a cada aeropuerto que hoy depende de una sola (705 aeropuertos) y se repite el ataque. "
                "Las rutas más probables conectan con hubs y no ayudan; las que unen aeropuertos pequeños de otras comunidades sí.")
    wr = D["whatif_rutas"]
    piv = wr.pivot_table(index="rutas nuevas", columns=["ataque", "pct"], values="gigante", sort=False).round(3)
    st.dataframe(piv, width="stretch")
    st.caption("Componente gigante (fracción de la red) tras remover el 5% o el 10% de aeropuertos con cada ataque.")
    st.markdown("**Cuántas rutas hacen falta** (propuesta «otra comunidad y sin hubs», priorizando por la intermediación del hub del que cuelgan):")
    st.dataframe(D["curva_K"].set_index("rutas nuevas (K)"), width="stretch")
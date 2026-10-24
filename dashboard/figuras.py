"""Funciones que construyen todas las figuras de Plotly del dashboard con un tema oscuro común."""

import numpy as np  # Cálculo numérico
import plotly.graph_objects as go  # Construcción de figuras de bajo nivel

# ---------------------------------------------------------------------------
# Paleta (pasos oscuros validados de la paleta categórica de referencia)
# ---------------------------------------------------------------------------
AZUL, NARANJA, AGUA, AMARILLO, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"  # Ranuras 1–5 en orden fijo
LEGITIMA, FRAUDE = AZUL, "#e66767"  # Colores de clase: azul = legítima, rojo = fraude
TEXTO, TEXTO_2, TEXTO_3 = "#f0f3fa", "#b4bdd0", "#7f8aa3"  # Tintas de texto (nunca el color de la serie)
REJILLA, SUPERFICIE = "rgba(255,255,255,0.06)", "#141a27"  # Rejilla recesiva y color de superficie (anillos de separación)
COLORES_MODELOS = {"LogisticRegression": AZUL, "RandomForest": NARANJA, "KNN": AGUA,  # Color fijo por modelo (sigue a la entidad, no al ranking)
                   "GradientBoosting": AMARILLO, "RVM": MAGENTA}
ETIQUETAS_MODELOS = {"LogisticRegression": "LogisticRegression", "RandomForest": "RandomForestClassifier",  # Nombre visible de cada modelo
                     "KNN": "KNeighborsClassifier", "GradientBoosting": "GradientBoostingClassifier",
                     "RVM": "RVM-Fraude (original)"}
SECUENCIAL = [[0, "#141a27"], [0.15, "#104281"], [0.4, "#256abf"], [0.7, "#5598e7"], [1, "#b7d3f6"]]  # Rampa azul de una sola tonalidad (oscuro → claro)
DIVERGENTE = [[0, "#3987e5"], [0.5, "#2a3448"], [1, "#e66767"]]  # Divergente: azul (negativo) – gris neutro – rojo (positivo)

ETIQUETAS = {  # Nombre legible de cada variable para ejes y desplegables
    "transaction_amount": "Monto de la transacción (USD)", "transaction_hour": "Hora de la transacción",
    "card_present": "Tarjeta presente", "customer_age": "Edad del cliente", "account_age_days": "Antigüedad de la cuenta (días)",
    "credit_limit": "Cupo de crédito", "credit_score": "Puntaje crediticio", "num_cards_held": "Número de tarjetas",
    "transactions_last_1h": "Transacciones última hora", "transactions_last_24h": "Transacciones últimas 24 h",
    "transactions_last_7d": "Transacciones últimos 7 días", "amount_last_1h": "Monto última hora",
    "amount_last_24h": "Monto últimas 24 h", "days_since_last_transaction": "Días desde la última transacción",
    "distance_from_home_km": "Distancia al domicilio (km)", "distance_from_last_tx_km": "Distancia a la transacción anterior (km)",
    "international_transaction": "Transacción internacional", "high_risk_country": "País de alto riesgo",
    "amount_vs_avg_ratio": "Monto / promedio del cliente", "used_chip": "Usó chip", "used_pin": "Usó PIN",
    "is_weekend": "Fin de semana", "avg_transaction_amount_30d": "Gasto promedio 30 días",
    "std_transaction_amount_30d": "Desviación del gasto 30 días", "num_declined_last_30d": "Rechazos últimos 30 días",
    "num_chargebacks_ever": "Contracargos históricos", "merchant_category": "Categoría del comercio",
    "merchant_country": "País del comercio", "transaction_day_of_week": "Día de la semana", "customer_gender": "Género del cliente",
    "customer_state": "Estado del cliente", "transaction_channel": "Canal", "device_type": "Dispositivo",
    "browser": "Navegador", "email_domain": "Dominio del correo",
}
DIAS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Orden natural de los días
DIAS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]  # Abreviaturas en español
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]  # Meses en español


def pct(x, dec=1):  # Formatea una proporción como porcentaje con coma decimal
    return f"{100 * x:.{dec}f} %".replace(".", ",")  # 0.0205 → "2,1 %"


def tema(fig, alto=360, leyenda=True):  # Aplica el tema oscuro común a una figura
    fig.update_layout(
        height=alto,  # Alto en píxeles
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",  # Fondo transparente: se ve la tarjeta
        font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXTO_2, size=12.5),  # Tipografía y color del texto
        margin=dict(l=10, r=10, t=30 if leyenda else 10, b=10),  # Márgenes ajustados
        hoverlabel=dict(bgcolor="#1c2435", bordercolor="#3a4660", font=dict(color=TEXTO, size=12.5)),  # Globo de información oscuro
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_2)),  # Leyenda arriba en fila
        showlegend=leyenda,  # Muestra u oculta la leyenda
        bargap=0.25,  # Separación entre barras
    )
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2a3448", tickfont=dict(color=TEXTO_3), title_font=dict(color=TEXTO_2), automargin=True)  # Eje X recesivo
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2a3448", tickfont=dict(color=TEXTO_3), title_font=dict(color=TEXTO_2), automargin=True)  # Eje Y recesivo
    return fig  # Devuelve la figura tematizada


def barras_redondeadas(fig):  # Esquinas redondeadas de 4 px y anillo de superficie entre barras
    fig.update_traces(marker_cornerradius=4, marker_line=dict(color=SUPERFICIE, width=2), selector=dict(type="bar"))  # Aplica a todas las barras
    return fig  # Devuelve la figura


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def fig_tipos_fraude(df):  # Composición de la variable objetivo por tipo de fraude
    conteo = df.loc[df["is_fraud"] == 1, "fraud_type"].value_counts().sort_values()  # Fraudes por tipo, de menor a mayor
    fig = go.Figure(go.Bar(x=conteo.values, y=conteo.index, orientation="h", marker_color=FRAUDE,  # Barras horizontales rojas
                           text=[f"{v:,} ({pct(v / conteo.sum(), 0)})".replace(",", ".", 1) for v in conteo.values],  # Etiqueta: conteo y porcentaje
                           textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,  # Etiqueta fuera de la barra
                           hovertemplate="%{y}: %{x:,} fraudes<extra></extra>"))  # Globo de información
    fig.update_xaxes(title="Transacciones fraudulentas", range=[0, conteo.max() * 1.3])  # Espacio para las etiquetas
    return barras_redondeadas(tema(fig, 300, leyenda=False))  # Tema y barras redondeadas


def fig_tasa_categorica(df, var, max_cat=15):  # Tasa de fraude por categoría con la tasa global como referencia
    g = df.groupby(var)["is_fraud"].agg(["mean", "size", "sum"])  # Tasa, total y fraudes por categoría
    if var == "transaction_day_of_week":  # Los días se muestran en su orden natural
        g = g.reindex(DIAS)  # Lunes → domingo
        g.index = DIAS_ES  # Etiquetas en español
    elif var == "transaction_hour":  # Las horas en orden 0–23
        g = g.sort_index()  # Orden numérico
    else:  # El resto de mayor a menor tasa
        g = g.sort_values("mean", ascending=False).head(max_cat)  # Las categorías con más riesgo primero
    etiquetas = [{0: "No", 1: "Sí"}.get(i, i) if var in BINARIAS else str(i) for i in g.index]  # Binarias como No/Sí
    global_ = df["is_fraud"].mean()  # Tasa global de fraude
    colores = [FRAUDE if m > global_ else "#5b6680" for m in g["mean"]]  # Rojo si supera la tasa global; gris si no
    fig = go.Figure(go.Bar(x=etiquetas, y=g["mean"], marker_color=colores, customdata=np.c_[g["size"], g["sum"]],  # Barras con datos extra
                           hovertemplate="%{x}<br>Tasa de fraude: %{y:.2%}<br>Transacciones: %{customdata[0]:,}<br>Fraudes: %{customdata[1]:,}<extra></extra>"))  # Globo
    fig.add_hline(y=global_, line=dict(color=TEXTO_3, dash="dot", width=1.5),  # Línea de la tasa global
                  annotation_text=f"Tasa global {pct(global_, 2)}", annotation_font=dict(color=TEXTO_2, size=11.5), annotation_position="top right")  # Etiqueta
    fig.update_yaxes(title="Tasa de fraude", tickformat=".0%")  # Eje Y en porcentaje
    fig.update_xaxes(title=ETIQUETAS.get(var, var), type="category")  # Eje X categórico
    return barras_redondeadas(tema(fig, 380, leyenda=False))  # Tema


BINARIAS = ["card_present", "international_transaction", "high_risk_country", "used_chip", "used_pin", "is_weekend"]  # Variables 0/1


def fig_numerica(df, var, tipo="histograma", log=False):  # Distribución de una variable numérica: fraude vs. legítima
    fig = go.Figure()  # Figura vacía
    for clase, nombre, color in [(0, "Legítima", LEGITIMA), (1, "Fraude", FRAUDE)]:  # Una serie por clase
        x = df.loc[df["is_fraud"] == clase, var]  # Valores de la clase
        if log:  # Escala logarítmica para variables muy sesgadas
            x = np.log10(x.clip(lower=0) + 1)  # log10(x + 1): conserva los ceros
        if tipo == "histograma":  # Histograma normalizado (las clases tienen tamaños muy distintos)
            fig.add_trace(go.Histogram(x=x, name=nombre, marker_color=color, opacity=0.6, histnorm="probability density", nbinsx=60,  # Densidad
                                       hovertemplate=f"{nombre}<br>%{{x}}<br>Densidad: %{{y:.4f}}<extra></extra>"))  # Globo
        elif tipo == "violin":  # Violín con caja interna
            fig.add_trace(go.Violin(y=x, name=nombre, line_color=color, fillcolor=color, opacity=0.65, box_visible=True,  # Violín
                                    meanline_visible=True, points=False, spanmode="hard"))  # Línea de la media y sin puntos
        else:  # Diagrama de caja
            fig.add_trace(go.Box(y=x, name=nombre, marker_color=color, boxmean=True, boxpoints=False))  # Caja con media
    titulo = ETIQUETAS.get(var, var) + (" · log10(x + 1)" if log else "")  # Título del eje
    if tipo == "histograma":  # Ejes del histograma
        fig.update_layout(barmode="overlay")  # Histogramas superpuestos
        fig.update_xaxes(title=titulo)  # Variable en X
        fig.update_yaxes(title="Densidad")  # Densidad en Y
    else:  # Ejes de violín y caja
        fig.update_yaxes(title=titulo)  # Variable en Y
    return tema(fig, 380)  # Tema


def fig_mensual(df):  # Tasa de fraude y volumen por mes, con la división temporal marcada
    mes = df["transaction_datetime"].dt.month  # Mes de cada transacción
    g = df.groupby(mes)["is_fraud"].agg(["mean", "size"])  # Tasa y volumen por mes
    colores = [AZUL if m < 11 else AMARILLO for m in g.index]  # Azul = entrenamiento (ene–oct), amarillo = prueba (nov–dic)
    fig = go.Figure(go.Bar(x=[MESES_ES[m - 1] for m in g.index], y=g["mean"], marker_color=colores, customdata=g["size"],  # Barras por mes
                           hovertemplate="%{x}<br>Tasa de fraude: %{y:.2%}<br>Transacciones: %{customdata:,}<extra></extra>"))  # Globo
    fig.add_annotation(x="May", y=g["mean"].max() * 1.12, text="Entrenamiento · ene–oct", showarrow=False, font=dict(color=TEXTO_2, size=12))  # Rótulo del periodo de entrenamiento
    fig.add_annotation(x="Nov", y=g["mean"].max() * 1.12, text="Prueba · nov–dic", showarrow=False, xshift=30, font=dict(color=TEXTO_2, size=12))  # Rótulo del periodo de prueba
    fig.add_vline(x=9.5, line=dict(color=TEXTO_3, dash="dash", width=1.5))  # Línea vertical en el corte temporal
    fig.update_yaxes(title="Tasa de fraude", tickformat=".1%", range=[0, g["mean"].max() * 1.22])  # Eje Y en porcentaje
    return barras_redondeadas(tema(fig, 330, leyenda=False))  # Tema


def fig_hora_dia(df):  # Mapa de calor: tasa de fraude por hora y día de la semana
    t = df.pivot_table(index="transaction_day_of_week", columns="transaction_hour", values="is_fraud", aggfunc="mean").reindex(DIAS)  # Tabla día × hora
    fig = go.Figure(go.Heatmap(z=t.values, x=t.columns, y=DIAS_ES, colorscale=SECUENCIAL, xgap=2, ygap=2,  # Celdas separadas por 2 px
                               colorbar=dict(title=dict(text="Tasa", font=dict(color=TEXTO_2)), tickformat=".0%", tickfont=dict(color=TEXTO_3), outlinewidth=0),  # Barra de color
                               hovertemplate="%{y} · %{x}:00<br>Tasa de fraude: %{z:.2%}<extra></extra>"))  # Globo
    fig.update_xaxes(title="Hora del día", dtick=2, showgrid=False)  # Eje de horas
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Lunes arriba
    return tema(fig, 330, leyenda=False)  # Tema


def fig_nulos_browser(df):  # Los nulos de browser dependen del canal: no son errores
    g = df.groupby("transaction_channel")["browser"].apply(lambda s: (s == "None").mean()).sort_values()  # Proporción sin navegador por canal
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=[AZUL if v < 0.5 else "#5b6680" for v in g.values],  # Barras por canal
                           text=[pct(v, 0) for v in g.values], textposition="outside", cliponaxis=False, textfont=dict(color=TEXTO_2),  # Etiquetas
                           hovertemplate="%{y}: %{x:.0%} sin navegador<extra></extra>"))  # Globo
    fig.update_xaxes(title="Transacciones sin navegador (nulo → 'None')", tickformat=".0%", range=[0, 1.18])  # Eje en porcentaje
    return barras_redondeadas(tema(fig, 260, leyenda=False))  # Tema


def fig_correlacion_objetivo(df, num_cols):  # Correlación de cada variable numérica con el fraude (punto-biserial)
    c = df[num_cols + ["is_fraud"]].corr()["is_fraud"].drop("is_fraud").sort_values()  # Correlación de Pearson con el objetivo
    fig = go.Figure(go.Bar(x=c.values, y=[ETIQUETAS[v] for v in c.index], orientation="h",  # Barras horizontales
                           marker_color=[FRAUDE if v > 0 else LEGITIMA for v in c.values],  # Rojo = aumenta el fraude; azul = lo reduce
                           hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"))  # Globo
    fig.update_xaxes(title="Correlación con is_fraud")  # Título del eje
    return barras_redondeadas(tema(fig, 620, leyenda=False))  # Tema


def fig_matriz_correlacion(df, num_cols):  # Matriz de correlación entre predictoras numéricas
    c = df[num_cols].corr()  # Matriz de Pearson
    nombres = [ETIQUETAS[v] if len(ETIQUETAS[v]) <= 24 else ETIQUETAS[v][:23] + "…" for v in num_cols]  # Etiquetas legibles y cortas
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=DIVERGENTE, xgap=1, ygap=1,  # Escala divergente centrada en 0
                               colorbar=dict(tickfont=dict(color=TEXTO_3), outlinewidth=0),  # Barra de color
                               hovertemplate="%{y} vs. %{x}<br>r = %{z:.2f}<extra></extra>"))  # Globo
    fig.update_xaxes(showticklabels=False, showgrid=False)  # Sin etiquetas en X (se leen en el globo)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))  # Etiquetas en Y
    return tema(fig, 620, leyenda=False)  # Tema


# ---------------------------------------------------------------------------
# ML Models
# ---------------------------------------------------------------------------
def fig_fuga(fuga):  # Comparación con fuga vs. sin fuga
    fig = go.Figure()  # Figura vacía
    for clave, nombre, color in [("con_fuga", "Con fuga (fraud_type, fraud_amount_lost)", FRAUDE), ("sin_fuga", "Sin fuga (flujo correcto)", AZUL)]:  # Dos escenarios
        v = fuga[clave]  # Métricas del escenario
        fig.add_trace(go.Bar(x=["AUC-ROC", "AUC-PR"], y=[v["auc_roc"], v["auc_pr"]], name=nombre, marker_color=color,  # Barras agrupadas
                             text=[f"{v['auc_roc']:.3f}", f"{v['auc_pr']:.3f}"], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False))  # Valores
    fig.add_hline(y=fuga["azar"], line=dict(color=TEXTO_3, dash="dot"), annotation_text=f"AUC-PR del azar = {fuga['azar']:.3f}",  # Referencia del azar
                  annotation_font=dict(color=TEXTO_2, size=11), annotation_position="top left")  # Etiqueta
    fig.update_yaxes(range=[0, 1.12], title="Área bajo la curva")  # Eje Y
    fig.update_layout(barmode="group")  # Barras agrupadas
    return barras_redondeadas(tema(fig, 320))  # Tema


def fig_roc(res, seleccionado):  # Curvas ROC de todos los modelos en prueba
    fig = go.Figure()  # Figura vacía
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_3, dash="dot", width=1), name="Azar", hoverinfo="skip"))  # Diagonal del azar
    for n, r in res.items():  # Una curva por modelo
        activo = n == seleccionado  # El seleccionado se resalta
        fig.add_trace(go.Scatter(x=r["roc"][0], y=r["roc"][1], mode="lines", name=f"{ETIQUETAS_MODELOS[n]} · {r['test']['auc_roc']:.3f}",  # Curva con AUC en la leyenda
                                 line=dict(color=COLORES_MODELOS[n], width=3 if activo else 1.5), opacity=1 if activo else 0.45,  # Grosor según selección
                                 hovertemplate=f"{ETIQUETAS_MODELOS[n]}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>"))  # Globo
    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])  # Eje X
    fig.update_yaxes(title="Tasa de verdaderos positivos (recall)", range=[0, 1.02])  # Eje Y
    fig.update_layout(legend=dict(orientation="v", x=0.45, y=0.05, yanchor="bottom", font=dict(size=11)))  # Leyenda dentro, abajo a la derecha
    return tema(fig, 400)  # Tema


def fig_pr(res, seleccionado, base):  # Curvas precisión-recall en prueba
    fig = go.Figure()  # Figura vacía
    fig.add_hline(y=base, line=dict(color=TEXTO_3, dash="dot", width=1), annotation_text=f"Azar = {base:.3f}",  # Referencia: tasa de fraude
                  annotation_font=dict(color=TEXTO_2, size=11), annotation_position="top right")  # Etiqueta
    for n, r in res.items():  # Una curva por modelo
        activo = n == seleccionado  # El seleccionado se resalta
        fig.add_trace(go.Scatter(x=r["pr"][0], y=r["pr"][1], mode="lines", name=f"{ETIQUETAS_MODELOS[n]} · {r['test']['auc_pr']:.3f}",  # Curva con AUC-PR
                                 line=dict(color=COLORES_MODELOS[n], width=3 if activo else 1.5), opacity=1 if activo else 0.45,  # Grosor
                                 hovertemplate=f"{ETIQUETAS_MODELOS[n]}<br>Recall %{{x:.3f}} · Precisión %{{y:.3f}}<extra></extra>"))  # Globo
    fig.update_xaxes(title="Recall (fraudes detectados)", range=[0, 1])  # Eje X
    fig.update_yaxes(title="Precisión (alertas correctas)", range=[0, 1.02])  # Eje Y
    fig.update_layout(legend=dict(orientation="v", x=0.45, y=0.98, yanchor="top", font=dict(size=11)))  # Leyenda arriba a la derecha
    return tema(fig, 400)  # Tema


def fig_confusion(m):  # Matriz de confusión anotada
    m = np.array(m)  # [[VN, FP], [FN, VP]]
    texto = [[f"VN<br><b>{m[0, 0]:,}</b>", f"FP<br><b>{m[0, 1]:,}</b>"], [f"FN<br><b>{m[1, 0]:,}</b>", f"VP<br><b>{m[1, 1]:,}</b>"]]  # Texto de cada celda
    fila = m / m.sum(axis=1, keepdims=True)  # Proporción por fila (normaliza el desbalance para el color)
    fig = go.Figure(go.Heatmap(z=fila, x=["Predicho: legítima", "Predicho: fraude"], y=["Real: legítima", "Real: fraude"],  # Celdas
                               text=texto, texttemplate="%{text}", textfont=dict(color=TEXTO, size=15), showscale=False, zmin=0, zmax=1,  # Texto y escala 0–1
                               colorscale=[[0, "#141a27"], [0.5, "#184f95"], [1, "#2a78d6"]],  # Rampa azul que conserva el contraste del texto blanco
                               xgap=3, ygap=3, hovertemplate="%{y} · %{x}<br>%{z:.1%} de la fila<extra></extra>"))  # Separación y globo
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Real legítima arriba
    fig.update_xaxes(showgrid=False, side="top")  # Etiquetas arriba
    return tema(fig, 330, leyenda=False)  # Tema


def fig_sobreajuste(res):  # AUC-ROC en train, validación cruzada, validación y prueba por modelo
    fig = go.Figure()  # Figura vacía
    nombres = [ETIQUETAS_MODELOS[n].replace(" (original)", "") for n in res]  # Etiquetas de los modelos
    for clave, nombre, color in [("train_auc_roc", "Train", "#5b6680"), ("cv_auc_roc", "Validación cruzada", AZUL),  # Series en orden fijo
                                 ("val_auc_roc", "Validación (umbral)", AGUA), ("test", "Prueba nov–dic", MAGENTA)]:
        y = [r["test"]["auc_roc"] if clave == "test" else r[clave] for r in res.values()]  # Valor de cada modelo
        fig.add_trace(go.Bar(x=nombres, y=y, name=nombre, marker_color=color, hovertemplate="%{x}<br>" + nombre + ": %{y:.3f}<extra></extra>"))  # Barras
    fig.update_yaxes(title="AUC-ROC", range=[0.6, 1.0])  # Eje truncado para ver las diferencias (se indica en la nota)
    fig.update_layout(barmode="group")  # Barras agrupadas
    return barras_redondeadas(tema(fig, 380))  # Tema


def fig_importancia(imp, color, top=15):  # Importancia por permutación (caída de AUC al desordenar cada variable)
    items = list(imp.items())[:top][::-1]  # Las más importantes, en orden ascendente para la barra horizontal
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[ETIQUETAS.get(k, k) for k, _ in items], orientation="h", marker_color=color,  # Barras
                           hovertemplate="%{y}<br>Caída del AUC: %{x:.4f}<extra></extra>"))  # Globo
    fig.update_xaxes(title="Caída media del AUC-ROC al permutar")  # Eje X
    return barras_redondeadas(tema(fig, 440, leyenda=False))  # Tema


def fig_recall_tipo(res, seleccionado):  # Recall por tipo de fraude (fraud_type solo para análisis de errores)
    tipos = list(next(iter(res.values()))["recall_por_tipo"].keys())  # Tipos de fraude
    fig = go.Figure()  # Figura vacía
    for n, r in res.items():  # Una serie por modelo
        fig.add_trace(go.Bar(x=tipos, y=[r["recall_por_tipo"][t] for t in tipos], name=ETIQUETAS_MODELOS[n], marker_color=COLORES_MODELOS[n],  # Barras
                             opacity=1 if n == seleccionado else 0.4, hovertemplate=ETIQUETAS_MODELOS[n] + "<br>%{x}: %{y:.1%}<extra></extra>"))  # Globo
    fig.update_yaxes(title="Recall con el umbral elegido", tickformat=".0%", range=[0, 1])  # Eje Y
    fig.update_layout(barmode="group", legend=dict(font=dict(size=10.5)))  # Agrupadas
    return barras_redondeadas(tema(fig, 380))  # Tema


def fig_delong(delong):  # Gráfico de bosque: diferencia de AUC del GA²M contra cada modelo, con IC 95 %
    nombres = list(delong.keys())[::-1]  # Modelos (orden invertido para leer de arriba abajo)
    dif = [delong[n]["dif"] for n in nombres]  # Diferencias
    lo = [delong[n]["dif"] - delong[n]["ic"][0] for n in nombres]  # Distancia al límite inferior
    hi = [delong[n]["ic"][1] - delong[n]["dif"] for n in nombres]  # Distancia al límite superior
    colores = [MAGENTA if delong[n]["p"] < 0.05 else "#7f8aa3" for n in nombres]  # Magenta si es significativa
    fig = go.Figure(go.Scatter(x=dif, y=[ETIQUETAS_MODELOS[n] for n in nombres], mode="markers",  # Puntos
                               marker=dict(size=13, color=colores, line=dict(color=SUPERFICIE, width=2)),  # Marcador con anillo
                               error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=TEXTO_3, thickness=2, width=6),  # Barras de error
                               customdata=[[delong[n]["p"], delong[n]["ic"][0], delong[n]["ic"][1]] for n in nombres],  # Datos del globo
                               hovertemplate="RVM vs. %{y}<br>ΔAUC = %{x:+.4f}<br>IC 95 %: [%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]<br>p = %{customdata[0]:.2g}<extra></extra>"))  # Globo
    fig.add_vline(x=0, line=dict(color=TEXTO_3, dash="dot"))  # Sin diferencia
    fig.update_xaxes(title="ΔAUC-ROC = RVM − modelo (prueba nov–dic)", zeroline=False)  # Eje X
    return tema(fig, 300, leyenda=False)  # Tema


def fig_medidor(p, umbral):  # Medidor de la probabilidad de fraude del simulador
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number=dict(suffix=" %", valueformat=".1f", font=dict(color=TEXTO, size=40)),  # Número central
        gauge=dict(axis=dict(range=[0, 100], tickcolor=TEXTO_3, tickfont=dict(color=TEXTO_3)),  # Escala 0–100 %
                   bar=dict(color=FRAUDE if p >= umbral else AZUL, thickness=0.3),  # Barra roja si supera el umbral
                   bgcolor="rgba(255,255,255,0.04)", borderwidth=0,  # Fondo del medidor
                   threshold=dict(line=dict(color=TEXTO, width=3), thickness=0.85, value=100 * umbral)),  # Marca del umbral
    ))
    return tema(fig, 240, leyenda=False)  # Tema


def fig_experimentos(exps):  # Gráfico de bosque: ΔAUC-ROC de cada modelo de la literatura frente a GradientBoosting
    nombres = [e["nombre"] for e in exps][::-1]  # Experimentos (de arriba abajo en el orden de ejecución)
    filas = {e["nombre"]: e for e in exps}  # Acceso por nombre
    dif = [filas[n]["dif"] for n in nombres]  # Diferencias de AUC
    lo = [filas[n]["dif"] - filas[n]["ic"][0] for n in nombres]  # Distancia al límite inferior del IC 95 %
    hi = [filas[n]["ic"][1] - filas[n]["dif"] for n in nombres]  # Distancia al límite superior
    colores = [MAGENTA if (filas[n]["p"] < 0.05 and filas[n]["dif"] > 0) else "#7f8aa3" for n in nombres]  # Magenta solo si supera significativamente
    fig = go.Figure(go.Scatter(x=dif, y=nombres, mode="markers",  # Puntos
                               marker=dict(size=13, color=colores, line=dict(color=SUPERFICIE, width=2)),  # Marcador con anillo de superficie
                               error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=TEXTO_3, thickness=2, width=6),  # Intervalos
                               customdata=[[filas[n]["test_auc"], filas[n]["p"]] for n in nombres],  # Datos del globo
                               hovertemplate="%{y}<br>AUC-ROC prueba = %{customdata[0]:.4f}<br>ΔAUC vs. GB = %{x:+.4f}<br>p = %{customdata[1]:.2g}<extra></extra>"))  # Globo
    fig.add_vline(x=0, line=dict(color=TEXTO_3, dash="dot"))  # Sin diferencia frente a GradientBoosting
    fig.update_xaxes(title="ΔAUC-ROC = modelo − GradientBoostingClassifier (prueba nov–dic)")  # Eje X
    return tema(fig, 340, leyenda=False)  # Tema



# ---------------------------------------------------------------------------
# EDA: valores atípicos
# ---------------------------------------------------------------------------
def tabla_atipicos(df, cols):  # Proporción de atípicos (regla de Tukey, 1,5 × IQR) y tasa de fraude dentro y fuera de ellos
    filas = []  # Una fila por variable
    for c in cols:  # Variables continuas
        q1, q3 = df[c].quantile([0.25, 0.75])  # Primer y tercer cuartil
        iqr = q3 - q1  # Rango intercuartílico
        atipico = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)  # Regla de Tukey
        filas.append({"variable": c, "pct_atipicos": float(atipico.mean()),  # Proporción de atípicos
                      "tasa_atipicos": float(df.loc[atipico, "is_fraud"].mean()) if atipico.any() else 0.0,  # Tasa de fraude en atípicos
                      "tasa_normales": float(df.loc[~atipico, "is_fraud"].mean())})  # Tasa de fraude en el resto
    return sorted(filas, key=lambda f: f["pct_atipicos"], reverse=True)  # De más a menos atípicos


def fig_atipicos(filas):  # Tasa de fraude en atípicos frente al resto, por variable
    filas = [f for f in filas if f["pct_atipicos"] > 0][::-1]  # Solo variables con atípicos (orden ascendente para barras horizontales)
    nombres = [ETIQUETAS[f["variable"]] for f in filas]  # Etiquetas legibles
    fig = go.Figure()  # Figura vacía
    fig.add_trace(go.Bar(y=nombres, x=[f["tasa_normales"] for f in filas], orientation="h", name="Sin atípicos", marker_color=LEGITIMA,  # Tasa en valores normales
                         hovertemplate="%{y}<br>Tasa de fraude sin atípicos: %{x:.2%}<extra></extra>"))
    fig.add_trace(go.Bar(y=nombres, x=[f["tasa_atipicos"] for f in filas], orientation="h", name="Atípicos (1,5 × IQR)", marker_color=FRAUDE,  # Tasa en atípicos
                         customdata=[f["pct_atipicos"] for f in filas],
                         hovertemplate="%{y}<br>Tasa de fraude en atípicos: %{x:.2%}<br>Atípicos: %{customdata:.1%} de las filas<extra></extra>"))
    fig.update_layout(barmode="group")  # Barras agrupadas
    fig.update_xaxes(title="Tasa de fraude", tickformat=".0%")  # Eje en porcentaje
    return barras_redondeadas(tema(fig, 520))  # Tema


# ---------------------------------------------------------------------------
# Diseño experimental: partición de los datos
# ---------------------------------------------------------------------------
def fig_particion(n):  # Esquema de la partición temporal y de validación
    fig = go.Figure()  # Figura vacía
    segmentos = [  # (fila, inicio, ancho, color, texto, ayuda)
        ("Datos", 0, 10, AZUL, f"Entrenamiento ene–oct · {n['train'] + n['val']:,} filas".replace(",", "."), "Periodo de entrenamiento"),
        ("Datos", 10, 2, AMARILLO, f"Prueba nov–dic · {n['test']:,}".replace(",", "."), "Se usa una sola vez"),
        ("Entrenamiento", 0, 8, AZUL, f"Ajuste + CV 5 pliegues · {n['train']:,}".replace(",", "."), "GridSearchCV + StratifiedKFold"),
        ("Entrenamiento", 8, 2, AGUA, f"Validación · {n['val']:,}".replace(",", "."), "20 % estratificado (aleatorio): umbral de decisión"),
        ("Comparación del listado", 0, 8, "#5b6680", "Ajuste ene–ago", "Candidatos del listado"),
        ("Comparación del listado", 8, 2, AGUA, "Elección sep–oct", "Configuración elegida sin mirar la prueba"),
        ("Comparación del listado", 10, 2, AMARILLO, "Prueba", "DeLong vs. GradientBoosting"),
    ]
    for fila, ini, ancho, color, texto, ayuda in segmentos:  # Dibuja cada segmento como barra horizontal
        fig.add_trace(go.Bar(y=[fila], x=[ancho], base=[ini], orientation="h", marker_color=color, text=[texto], textposition="inside",  # Barra
                             insidetextanchor="middle", textfont=dict(color=TEXTO, size=12), hovertemplate=f"{texto}<br>{ayuda}<extra></extra>", showlegend=False))
    fig.update_layout(barmode="overlay")  # Las barras se ubican por su base
    fig.update_xaxes(tickvals=[i + 0.5 for i in range(12)], ticktext=MESES_ES, range=[0, 12], title="Meses de 2026", showgrid=False)  # Eje de meses
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Primera fila arriba
    return barras_redondeadas(tema(fig, 260, leyenda=False))  # Tema


# ---------------------------------------------------------------------------
# Fundamentos de la RVM
# ---------------------------------------------------------------------------
def fig_rvm_relevantes(historial):  # Número de funciones base relevantes por iteración (la poda ARD)
    it = [h["iteracion"] for h in historial]  # Iteraciones
    fig = go.Figure(go.Scatter(x=it, y=[h["relevantes"] for h in historial], mode="lines", line=dict(color=MAGENTA, width=2),  # Curva
                               hovertemplate="Iteración %{x}<br>Funciones relevantes: %{y}<extra></extra>"))
    fig.update_xaxes(title="Iteración (Laplace → evidencia)")  # Eje X
    fig.update_yaxes(title="Funciones base relevantes")  # Eje Y
    return tema(fig, 300, leyenda=False)  # Tema


def fig_rvm_evidencia(historial):  # Log-evidencia aproximada por iteración (lo que maximiza la RVM)
    it = [h["iteracion"] for h in historial]  # Iteraciones
    fig = go.Figure(go.Scatter(x=it, y=[h["log_evidencia"] for h in historial], mode="lines", line=dict(color=AZUL, width=2),  # Curva
                               hovertemplate="Iteración %{x}<br>ln p(y | α) ≈ %{y:,.1f}<extra></extra>"))
    fig.update_xaxes(title="Iteración (Laplace → evidencia)")  # Eje X
    fig.update_yaxes(title="Log-evidencia ln p(y | α)")  # Eje Y
    return tema(fig, 300, leyenda=False)  # Tema


def fig_rvm_alphas(alphas, umbral=6.0):  # Precisión ARD de cada función base: las de α grande se podan
    la = np.array(alphas["log10_alpha"])  # log10 α
    rel = np.array(alphas["relevante"])  # ¿Sobrevivió a la poda?
    var = np.array(alphas["variable"])  # Variable de origen
    fig = go.Figure()  # Figura vacía
    for flag, nombre, color in [(True, "Relevante (se conserva)", MAGENTA), (False, "Podada (α → ∞, peso 0)", "#5b6680")]:  # Dos grupos
        m = rel == flag  # Máscara del grupo
        fig.add_trace(go.Scatter(x=np.flatnonzero(m), y=np.minimum(la[m], 12), mode="markers", name=nombre,  # Puntos
                                 marker=dict(size=8, color=color, line=dict(color=SUPERFICIE, width=1)),
                                 customdata=[ETIQUETAS.get(v, v) for v in var[m]],
                                 hovertemplate="Función base %{x}<br>Variable: %{customdata}<br>log10 α = %{y:.2f}<extra></extra>"))
    fig.add_hline(y=umbral, line=dict(color=TEXTO_3, dash="dot"), annotation_text="umbral de poda α = 10⁶",  # Umbral de poda
                  annotation_font=dict(color=TEXTO_2, size=11), annotation_position="top left")
    fig.update_xaxes(title="Función base (splines y one-hot, agrupadas por variable)")  # Eje X
    fig.update_yaxes(title="log10 α (precisión del prior)")  # Eje Y
    return tema(fig, 340)  # Tema


def fig_rvm_relevancia(curvas, relevancia, top=20):  # Rango de contribución al logit de cada variable (su efecto total en la RVM)
    rango = {v: (max(c["y"]) - min(c["y"])) if c["y"] else 0.0 for v, c in curvas.items()}  # Máximo − mínimo de la curva
    items = sorted(rango.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]  # Las más influyentes
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[ETIQUETAS[k] for k, _ in items], orientation="h", marker_color=MAGENTA,  # Barras
                           customdata=[[relevancia[k]["relevantes"], relevancia[k]["bases"]] for k, _ in items],
                           hovertemplate="%{y}<br>Rango de contribución: %{x:.2f}<br>Funciones relevantes: %{customdata[0]} de %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title="Rango de contribución al logit (máx − mín)")  # Eje X
    return barras_redondeadas(tema(fig, 520, leyenda=False))  # Tema


def fig_rvm_curva(curva, var):  # Contribución de una variable al logit con banda de incertidumbre posterior (± 1,96 σ)
    y, sd = np.array(curva["y"]), np.array(curva["sd"])  # Media y desviación posterior
    if curva["tipo"] == "num":  # Continua: curva suave con banda
        x = list(curva["x"])  # Valores de la variable
        fig = go.Figure([
            go.Scatter(x=x + x[::-1], y=list(y + 1.96 * sd) + list((y - 1.96 * sd)[::-1]), fill="toself", fillcolor="rgba(213,81,129,0.18)",  # Banda del 95 %
                       line=dict(width=0), hoverinfo="skip", name="IC 95 % posterior"),
            go.Scatter(x=x, y=y, mode="lines", line=dict(color=MAGENTA, width=2.5), name="Contribución media",  # Curva media
                       hovertemplate="x = %{x:,.2f}<br>Contribución al logit: %{y:+.3f}<extra></extra>"),
        ])
    else:  # Discreta o categórica: barras con barras de error
        x = [("Sí" if v == 1 else "No") if var in BINARIAS else (f"{v:g}" if isinstance(v, float) else str(v)) for v in curva["x"]]  # Etiquetas
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=[FRAUDE if v > 0 else LEGITIMA for v in y], name="Contribución",  # Barras
                               error_y=dict(type="data", array=1.96 * sd, color=TEXTO_3, thickness=1.5, width=4),  # IC 95 %
                               hovertemplate="%{x}<br>Contribución al logit: %{y:+.3f}<extra></extra>"))
        fig.update_xaxes(type="category")  # Eje categórico
        barras_redondeadas(fig)  # Barras redondeadas
    fig.add_hline(y=0, line=dict(color=TEXTO_3, width=1))  # Contribución nula
    fig.update_xaxes(title=ETIQUETAS.get(var, var))  # Eje X
    fig.update_yaxes(title="Contribución al logit de fraude")  # Eje Y
    return tema(fig, 380, leyenda=curva["tipo"] == "num")  # Tema



# ---------------------------------------------------------------------------
# Tiempos de cómputo y pruebas de significancia
# ---------------------------------------------------------------------------
def fig_tiempos(tiempos):  # Tiempo de GridSearchCV completo y de un solo ajuste por modelo (misma unidad: segundos, escala log)
    nombres = list(tiempos)  # Modelos
    etq = [ETIQUETAS_MODELOS[n].replace(" (original)", "") for n in nombres]  # Etiquetas
    fig = go.Figure()  # Figura vacía
    fig.add_trace(go.Bar(x=etq, y=[tiempos[n]["grid_s"] for n in nombres], name="GridSearchCV completo (5 pliegues × malla + reajuste)", marker_color=AZUL,  # Búsqueda
                         text=[f"{tiempos[n]['grid_s']:.0f} s" for n in nombres], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,
                         hovertemplate="%{x}<br>GridSearchCV: %{y:.1f} s<extra></extra>"))
    fig.add_trace(go.Bar(x=etq, y=[tiempos[n]["fit_s"] for n in nombres], name="Un ajuste con los mejores hiperparámetros", marker_color=MAGENTA,  # Ajuste único
                         text=[f"{tiempos[n]['fit_s']:.1f} s" for n in nombres], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,
                         customdata=[tiempos[n]["n_fit"] for n in nombres],
                         hovertemplate="%{x}<br>Un ajuste: %{y:.2f} s con %{customdata:,} filas<extra></extra>"))
    fig.update_yaxes(title="Segundos (escala logarítmica)", type="log")  # Escala log: los tiempos difieren en órdenes de magnitud
    fig.update_layout(barmode="group")  # Barras agrupadas
    return barras_redondeadas(tema(fig, 380))  # Tema


def fig_prediccion(tiempos):  # Latencia de predicción por cada 1 000 transacciones
    nombres = sorted(tiempos, key=lambda n: tiempos[n]["pred_ms_1000"])  # De más rápido a más lento
    fig = go.Figure(go.Bar(y=[ETIQUETAS_MODELOS[n] for n in nombres], x=[tiempos[n]["pred_ms_1000"] for n in nombres], orientation="h",  # Barras
                           marker_color=[COLORES_MODELOS[n] for n in nombres], text=[f"{tiempos[n]['pred_ms_1000']:.1f} ms" for n in nombres],
                           textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:.2f} ms por 1 000 transacciones<extra></extra>"))
    fig.update_xaxes(title="Milisegundos por 1 000 transacciones (escala logarítmica)", type="log")  # Escala log
    return barras_redondeadas(tema(fig, 300, leyenda=False))  # Tema


def fig_nemenyi(friedman, titulo_metrica):  # Diagrama de diferencia crítica (Demšar, 2006): rango medio de cada modelo y la DC de Nemenyi
    rangos = friedman["rangos"]  # Rango medio por modelo (1 = mejor)
    nombres = sorted(rangos, key=rangos.get)  # Del mejor al peor
    cd = friedman["cd"]  # Diferencia crítica
    mejor = rangos[nombres[0]]  # Rango del mejor modelo
    fig = go.Figure()  # Figura vacía
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.5, y1=len(nombres) - 0.5, fillcolor="rgba(213,81,129,0.10)", line=dict(width=0))  # Zona sin diferencia con el mejor
    fig.add_trace(go.Scatter(x=[rangos[n] for n in nombres], y=[ETIQUETAS_MODELOS[n] for n in nombres], mode="markers",  # Puntos
                             marker=dict(size=14, color=[COLORES_MODELOS[n] for n in nombres], line=dict(color=SUPERFICIE, width=2)),
                             error_x=dict(type="constant", value=cd / 2, color=TEXTO_3, thickness=2, width=6),  # Media DC a cada lado
                             hovertemplate="%{y}<br>Rango medio: %{x:.2f}<extra></extra>"))
    fig.add_annotation(x=mejor + cd, y=len(nombres) - 0.5, text=f"DC de Nemenyi = {cd:.2f}".replace(".", ","), showarrow=False,  # Rótulo de la DC
                       xanchor="right", yanchor="bottom", font=dict(color=TEXTO_2, size=11))
    fig.update_xaxes(title=f"Rango medio en 5×2cv, {titulo_metrica} (1 = mejor)", range=[0.5, len(nombres) + 0.5])  # Eje X
    fig.update_yaxes(autorange="reversed")  # Mejor arriba
    return tema(fig, 300, leyenda=False)  # Tema

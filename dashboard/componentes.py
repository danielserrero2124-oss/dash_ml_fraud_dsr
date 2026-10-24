"""Piezas reutilizables de la interfaz: tarjetas, indicadores, tablas, fórmulas y etiquetas de la rúbrica."""

from dash import dcc, html  # Componentes de Dash

from dashboard import figuras as fg  # Etiquetas legibles de las variables

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}  # Barra de herramientas simplificada
RUBRICA = {  # Criterios de la rúbrica de sustentación: (nombre, ponderación, minutos)
    1: ("Problema y formulación matemática", 10, 2), 2: ("EDA, datos y preprocesamiento", 15, 3),
    3: ("Fundamentos teóricos del modelo", 25, 5), 4: ("Implementación y diseño experimental", 15, 3),
    5: ("Resultados y evaluación crítica", 15, 3), 6: ("Conclusiones y limitaciones", 10, 2),
    7: ("Dominio teórico y respuestas a preguntas", 10, 2),
}


def numero(valor, dec=3):  # Formatea un número con coma decimal
    return f"{valor:.{dec}f}".replace(".", ",")  # 0.8612 → "0,861"


def miles(valor):  # Formatea enteros con punto de miles
    return f"{valor:,.0f}".replace(",", ".")  # 100000 → "100.000"


def tarjeta(*hijos, clase=""):  # Contenedor con fondo, borde y sombra
    return html.Div(list(hijos), className=f"tarjeta {clase}".strip())  # div con la clase CSS "tarjeta"


def kpi(etiqueta, valor, detalle="", id_valor=None, id_detalle=None):  # Indicador numérico
    cifra = html.Div(valor, className="valor", **({"id": id_valor} if id_valor else {}))  # Cifra grande (con id si cambia por callback)
    nota = html.Div(detalle, className="detalle", **({"id": id_detalle} if id_detalle else {}))  # Aclaración (con id opcional)
    return tarjeta(html.Div(etiqueta, className="etiqueta"), cifra, nota, clase="kpi")  # Etiqueta + cifra + detalle


def desplegable(id_, opciones, valor):  # Lista desplegable con el estilo del tema
    return dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False)  # Siempre con una opción elegida


def grafico(id_, figura=None):  # Gráfico de Plotly con la configuración común
    return dcc.Graph(id=id_, figure=figura or {}, config=CONFIG)  # Figura fija o actualizada por callback


def opciones_vars(variables):  # Opciones de desplegable con el nombre legible de cada variable
    return [{"label": fg.ETIQUETAS[v], "value": v} for v in variables]  # Texto visible → nombre de columna


def pasos(lista):  # Lista de pasos numerados
    return html.Div([html.Div([html.Div(str(i), className="paso-numero"), html.Div(t, className="paso-texto")], className="paso")  # Número + texto
                     for i, t in enumerate(lista, 1)])  # Un paso por elemento


def tabla(encabezados, filas, numericas=()):  # Tabla HTML con el estilo del tema; numericas = índices de columnas alineadas a la derecha
    cab = html.Thead(html.Tr([html.Th(h, className="num" if i in numericas else "") for i, h in enumerate(encabezados)]))  # Encabezado
    cuerpo = html.Tbody([html.Tr([html.Td(c, className="num" if i in numericas else "") for i, c in enumerate(f)]) for f in filas])  # Filas
    return html.Div(html.Table([cab, cuerpo], className="tabla"), className="tabla-envoltura")  # Tabla con desplazamiento horizontal


def formula(texto):  # Bloque de texto con fórmulas en LaTeX (MathJax)
    return dcc.Markdown(texto, mathjax=True, className="formula")  # $...$ en línea y $$...$$ en bloque


def rubrica(n):  # Título de la parte de la rúbrica que cubre la sección
    return html.Div(html.Span(RUBRICA[n][0], className="rubrica-nombre"), className="rubrica")  # Solo el título del criterio


def pregunta(titulo, *respuesta):  # Pregunta desplegable (para la sección de preguntas de la sustentación)
    return html.Details([html.Summary(titulo), html.Div(list(respuesta), className="respuesta")], className="pregunta")  # Acordeón nativo de HTML


def cita(texto, url=None):  # Referencia bibliográfica con enlace opcional
    return html.Div([texto, " ", html.A("DOI", href=url, target="_blank")] if url else [texto], className="cita")  # Bloque con borde de acento

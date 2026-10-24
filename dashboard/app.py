"""Dashboard interactivo del proyecto de detección de fraude en tarjetas de crédito (Dash + Plotly).

Secciones: Contexto · EDA · ML Models (con sub-pestañas organizadas según la rúbrica de sustentación).
Antes de abrirlo, entrenar los modelos una vez:  python -m dashboard.entrenar
Ejecución (desde la raíz del proyecto):           python -m dashboard.app   →   http://localhost:8050
"""

from dash import Dash, Input, Output, callback, dcc, html  # Aplicación, componentes y decorador de callbacks

app = Dash(__name__, title="Fraude en Tarjetas · ML Dashboard", suppress_callback_exceptions=True)  # Crea la aplicación (carga assets/ sola)
server = app.server  # Servidor Flask interno, necesario para desplegar con gunicorn

from dashboard.sec_contexto import seccion_contexto  # noqa: E402  Sección 1 (se importa después de crear la app)
from dashboard.sec_eda import seccion_eda  # noqa: E402  Sección 2 y sus callbacks
from dashboard.sec_modelos import seccion_modelos  # noqa: E402  Sección 3 y sus callbacks

app.layout = html.Div([
    html.Div([  # Encabezado
        html.Div([html.Div("🛡️", className="marca-icono"),  # Icono
                  html.Div([html.H1("Detección de Fraude en Tarjetas de Crédito"),  # Título
                            html.P("Proyecto integrador de aprendizaje automático · 100 000 transacciones de 2026")])], className="marca"),  # Subtítulo
        html.Div([html.Span("Dash + Plotly", className="chip"), html.Span("scikit-learn", className="chip"), html.Span("RVM bayesiana desde cero", className="chip")]),  # Etiquetas
    ], className="encabezado"),
    dcc.Tabs(id="pestanas", value="contexto", className="pestanas", children=[  # Navegación principal
        dcc.Tab(label="Contexto", value="contexto", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="EDA", value="eda", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="ML Models", value="ml", className="tab", selected_className="tab--selected"),
    ]),
    html.Div(id="contenido"),  # Contenido de la pestaña activa
    html.Div("Maestría en Machine Learning · Proyecto integrador · Datos sintéticos con fines académicos", className="pie"),  # Pie de página
], className="contenedor")


@callback(Output("contenido", "children"), Input("pestanas", "value"))  # Cambia de sección
def mostrar_seccion(pestana):
    return {"contexto": seccion_contexto, "eda": seccion_eda, "ml": seccion_modelos}[pestana]()  # Construye la sección elegida


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    app.run(debug=False, port=8050)  # Inicia el servidor en http://localhost:8050

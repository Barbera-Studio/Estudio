import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="IBEX vs PIB España", layout="wide")

st.title("📊 Evolución del IBEX 35 vs PIB de España")
st.markdown("""
### 📊 Análisis comparativo de indicadores económicos clave en España

Esta aplicación presenta una visualización interactiva de la evolución del índice bursátil **IBEX 35**, el **PIB nominal español**, la **inflación anual**, la **deuda pública como porcentaje del PIB**, y los **tipos de interés oficiales del BCE**.  
El objetivo es ofrecer una perspectiva integrada sobre el comportamiento de los mercados financieros en relación con los principales indicadores macroeconómicos del país.

**Cada sección incluye:**

📈 Gráficos temporales que muestran la evolución histórica de cada indicador desde el año 2000.  
📊 Tablas comparativas con valores normalizados para facilitar la comparación entre series de distinta magnitud.  
⬇️ Botones de descarga para exportar los datos en formato CSV.  
🎚️ Filtros por rango de años para personalizar el análisis según el período de interés.

Este enfoque permite observar **correlaciones**, **divergencias** y **patrones** entre el rendimiento del mercado bursátil y variables económicas fundamentales, facilitando el análisis económico, financiero y político de la situación española en distintos contextos históricos.
""")

# 1. Configuración inicial
start_date = "2000-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

with st.spinner("Descargando datos del IBEX 35..."):
    try:
        ticker = "^IBEX"
        df = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=False)
        df = df.reset_index()
        df["close_norm_100"] = df["Close"] / df["Close"].iloc[0] * 100
        ibex = df[["Date", "Close", "close_norm_100"]].copy()
        ibex.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        st.success("✅ Datos del IBEX 35 cargados correctamente.")
    except Exception as e:
        st.error(f"❌ Error al descargar datos del IBEX: {e}")
        st.stop()

# 2. PIB España
with st.spinner("Cargando datos del PIB español..."):
    try:
        df_macro = pd.read_csv("pib_esp.csv")
        df_macro["date"] = pd.to_datetime(df_macro["date"])
        df_macro.rename(columns={"value": "pib"}, inplace=True)
        df_macro["pib_norm_100"] = df_macro["pib"] / df_macro["pib"].iloc[0] * 100
        pib_esp = df_macro[["date", "pib", "pib_norm_100"]].copy()
        st.success("✅ PIB español cargado correctamente.")
    except Exception as e:
        st.warning(f"⚠️ Error al cargar PIB: {e}")
        pib_esp = pd.DataFrame()

# 3. Tipos BCE
with st.spinner("Cargando datos de tipos de interés..."):
    try:
        df_tipos = pd.read_csv("tipos_bce.csv")
        df_tipos.rename(columns={
            "DATE": "date",
            "Main refinancing operations - fixed rate tenders (fixed rate) (date of changes) - Level (FM.B.U2.EUR.4F.KR.MRR_FR.LEV)": "tipo"
        }, inplace=True)
        df_tipos["date"] = pd.to_datetime(df_tipos["date"])
        df_tipos["year"] = df_tipos["date"].dt.year
        df_tipos["tipos_norm_100"] = df_tipos["tipo"] / df_tipos["tipo"].iloc[0] * 100
        tipos_interes = df_tipos[["date", "year", "tipo", "tipos_norm_100"]]
        st.success("✅ Datos de tipos de interés cargados correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar datos de tipos de interés: {e}")
        st.stop()

# 4. IBEX anual
ibex["year"] = ibex["date"].dt.year
ibex_anual = ibex.groupby("year")["close_norm_100"].mean().reset_index()
ibex_anual["date"] = pd.to_datetime(ibex_anual["year"].astype(str) + "-12-31")

# 5. Unir IBEX + PIB
comparativa = pd.merge(ibex_anual, pib_esp, on="date", how="inner")

# 6. Inflación
inflacion_data = {
    2000: 3.4, 2001: 2.8, 2002: 3.5, 2003: 2.6, 2004: 3.1, 2005: 3.4, 2006: 3.6, 2007: 4.2,
    2008: 1.4, 2009: -0.3, 2010: 1.8, 2011: 3.1, 2012: 2.4, 2013: 1.4, 2014: -0.2, 2015: -0.5,
    2016: 1.6, 2017: 1.1, 2018: 1.7, 2019: 0.7, 2020: -0.5, 2021: 3.1, 2022: 5.7, 2023: 3.1, 2024: 2.8
}
df_inflacion = pd.DataFrame(list(inflacion_data.items()), columns=["year", "inflacion"])
df_inflacion["date"] = pd.to_datetime(df_inflacion["year"].astype(str) + "-12-31")
comparativa = pd.merge(comparativa, df_inflacion, on="date", how="left")

# 7. Filtro años
min_year = comparativa["date"].dt.year.min()
max_year = comparativa["date"].dt.year.max()
rango = st.slider("Selecciona el rango de años", min_year, max_year, (min_year, max_year), step=1)

start_str = f"{rango[0]}-01-01"
end_str = f"{rango[1]}-12-31"
tick0_str = f"{rango[0]}-12-31"

comparativa_filtrada = comparativa[
    (comparativa["date"] >= start_str) &
    (comparativa["date"] <= end_str)
]
tipos_interes_filtrados = tipos_interes[
    (tipos_interes["date"] >= start_str) &
    (tipos_interes["date"] <= end_str)
]

tabs = st.tabs(["📈 IBEX vs PIB", "💰 Deuda pública", "🏦 Tipos de interés"])

def responsive_style(start_str: str, end_str: str):
    span = (pd.to_datetime(end_str).year - pd.to_datetime(start_str).year) + 1
    if span <= 8:
        return dict(height=640, line_w=3, font=14, label_font=13, years_step=1, smooth=False, window=1)
    elif span <= 15:
        return dict(height=720, line_w=2.6, font=13, label_font=12, years_step=2, smooth=True, window=2)
    else:
        return dict(height=800, line_w=2.2, font=12, label_font=11, years_step=3, smooth=True, window=3)

def year_tickvals(start_str: str, end_str: str, step_years: int):
    y0 = pd.to_datetime(start_str).year
    y1 = pd.to_datetime(end_str).year
    years = list(range(y0, y1 + 1, step_years))
    return [f"{y}-12-31" for y in years], [str(y) for y in years]

def maybe_smooth(df: pd.DataFrame, col: str, window: int):
    if window <= 1:
        return df[col]
    return df[col].rolling(window=window, min_periods=1, center=True).mean()

# ===== Pestaña 1: IBEX vs PIB =====
with tabs[0]:
    st.markdown("<h3 style='margin:0 0 8px 0;'>Evolución IBEX 35 vs PIB España</h3>", unsafe_allow_html=True)

    sty = responsive_style(start_str, end_str)
    tickvals, ticktext = year_tickvals(start_str, end_str, sty["years_step"])

    if comparativa_filtrada.empty:
        st.warning("No hay datos para el rango seleccionado.")
    else:
        corr = comparativa_filtrada[["close_norm_100", "pib_norm_100"]].corr().iloc[0, 1]

        ibex_series = maybe_smooth(comparativa_filtrada, "close_norm_100", sty["window"])
        pib_series = maybe_smooth(comparativa_filtrada, "pib_norm_100", sty["window"])
        inf_series = maybe_smooth(comparativa_filtrada, "inflacion", sty["window"])

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.72, 0.28],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # Serie IBEX
        fig.add_trace(
            go.Scatter(
                x=comparativa_filtrada["date"],
                y=ibex_series,
                mode="lines",
                name="IBEX 35",
                line=dict(color="#2563eb", width=sty["line_w"]),
                hovertemplate="<b>IBEX</b>: %{y:.2f}<br>%{x|%Y}<extra></extra>",
                line_shape="spline" if sty["smooth"] else "linear",
            ),
            row=1,
            col=1,
        )

        # Serie PIB
        fig.add_trace(
            go.Scatter(
                x=comparativa_filtrada["date"],
                y=pib_series,
                mode="lines",
                name="PIB España",
                line=dict(color="#16a34a", width=sty["line_w"]),
                hovertemplate="<b>PIB</b>: %{y:.2f}<br>%{x|%Y}<extra></extra>",
                line_shape="spline" if sty["smooth"] else "linear",
            ),
            row=1,
            col=1,
        )

        # Serie inflación (fila 2)
        fig.add_trace(
            go.Scatter(
                x=comparativa_filtrada["date"],
                y=inf_series,
                mode="lines",
                name="Inflación (%)",
                line=dict(color="#92400e", width=max(1.6, sty["line_w"] - 0.4), dash="dash"),
                opacity=0.9,
                hovertemplate="<b>Inflación</b>: %{y:.2f}%<br>%{x|%Y}<extra></extra>",
                line_shape="spline" if sty["smooth"] else "linear",
            ),
            row=2,
            col=1,
        )

        # ===== EVENTOS: vrect + texto vertical abajo, centrado a cada franja =====
        eventos = [
            dict(ini="2008-01-01", fin="2010-01-01", label="Recesión 2008-10", color="#ef4444"),
            dict(ini="2020-01-01", fin="2021-01-01", label="COVID-19", color="#7c3aed"),
            dict(ini="2022-01-01", fin="2023-01-01", label="Post-COVID inflación", color="#ea580c"),
        ]

        span_years = (pd.to_datetime(end_str).year - pd.to_datetime(start_str).year) + 1
        if span_years <= 8:
            event_font_size = 16
            event_opacity = 0.20
        elif span_years <= 15:
            event_font_size = 14
            event_opacity = 0.18
        else:
            event_font_size = 12
            event_opacity = 0.16

        # Rango del eje superior y espacio para etiquetas
        y1_min = min(ibex_series.min(), pib_series.min())
        y1_max = max(ibex_series.max(), pib_series.max())
        margin_bottom = (y1_max - y1_min) * 0.10  # 10% del rango

        fig.update_yaxes(range=[y1_min - margin_bottom, y1_max + 3], row=1, col=1)

        # y en paper para los textos, alineados visualmente con la base de las barras
        # (más arriba que antes para que "se peguen" al pie de la franja)
        label_y_paper = 0.5  # subir/bajar milimétricamente aquí si hace falta

        for ev in eventos:
            ini_dt = pd.to_datetime(ev["ini"])
            fin_dt = pd.to_datetime(ev["fin"])
            x_center = ini_dt + (fin_dt - ini_dt) / 2

            # Banda
            fig.add_vrect(
                x0=ev["ini"],
                x1=ev["fin"],
                fillcolor=ev["color"],
                opacity=event_opacity,
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )

            # Texto vertical, centrado a la banda, abajo
            fig.add_annotation(
                x=x_center,
                y=label_y_paper,
                xref="x1",
                yref="paper",
                text=f"<b>{ev['label']}</b>",
                showarrow=False,
                font=dict(size=event_font_size),
                textangle=90,
                align="center",
            )

        # ===== FIN EVENTOS =====

        fig.add_annotation(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            text=f"Corr IBEX vs PIB: {corr:.2f}",
            showarrow=False,
            font=dict(size=sty["label_font"]),
        )

        fig.update_layout(
            hovermode="x unified",
            height=sty["height"],
            margin=dict(l=28, r=28, t=80, b=180),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=dict(size=sty["label_font"]),
            ),
        )

        fig.update_yaxes(title_text="Índice normalizado (Base 100)", row=1, col=1)
        fig.update_yaxes(title_text="Inflación (%)", row=2, col=1)

        # Rango de la inflación
        y2_min = inf_series.min()
        y2_max = inf_series.max()
        fig.update_yaxes(range=[y2_min - 0.5, y2_max + 0.5], row=2, col=1)

        fig.update_xaxes(
            title_text="Fecha",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            ticks="outside",
            ticklen=6,
            tickcolor="rgba(0,0,0,0.45)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            automargin=True,
            range=[start_str, end_str],
            row=2,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, title_text=None, row=1, col=1)
        fig.update_xaxes(title_text=None, row=2, col=1)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "modeBarButtonsToRemove": [
                    "toggleSpikelines",
                    "autoScale2d",
                    "lasso2d",
                    "select2d",
                ],
            },
        )

# ===== Pestaña 2: Deuda (placeholder) =====
with tabs[1]:
    st.info("Sección de deuda pública pendiente de implementar.")

# ===== Pestaña 3: Tipos BCE =====
with tabs[2]:
    if tipos_interes_filtrados.empty:
        st.warning("No hay datos de tipos de interés para ese rango.")
    else:
        fig4 = go.Figure(
            go.Scatter(
                x=tipos_interes_filtrados["date"],
                y=tipos_interes_filtrados["tipo"],
                mode="lines+markers",
                name="Tipo BCE (%)",
                line=dict(color="navy"),
                hovertemplate="%{y:.2f}%<br>%{x|%Y-%m-%d}",
            )
        )
        fig4.update_layout(
            title="Tipo de interés principal del BCE",
            xaxis_title="Fecha",
            yaxis_title="Porcentaje (%)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig4.update_xaxes(
            tickmode="linear",
            tick0=tick0_str,
            dtick="M36",
            tickformat="%Y",
            range=[start_str, end_str],
        )
        st.plotly_chart(fig4, use_container_width=True)

# Correlación disponible
if not comparativa_filtrada.empty:
    correlacion = comparativa_filtrada[["close_norm_100", "pib_norm_100"]].corr().iloc[0, 1]

# Datos descargables
with st.expander("📋 Ver datos comparativos IBEX vs PIB"):
    st.dataframe(comparativa_filtrada)
    csv = comparativa_filtrada.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV comparativa IBEX vs PIB",
        data=csv,
        file_name="comparativa_ibex_pib.csv",
        mime="text/csv",
    )

with st.expander("📋 Ver datos de tipos de interés"):
    st.dataframe(tipos_interes_filtrados)
    csv_tipos = tipos_interes_filtrados.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV de tipos de interés",
        data=csv_tipos,
        file_name="tipos_interes_bce.csv",
        mime="text/csv",
    )

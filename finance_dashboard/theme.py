import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
CHART_COLORS = ["#1e3a5f", "#2d6a9f", "#4a90a4", "#6b8cae", "#94a3b8"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=CHART_FONT, size=13, color="#1c2333"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORS,
        margin=dict(l=48, r=24, t=56, b=48),
        title=dict(font=dict(size=16, color="#1c2333")),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(28,35,51,0.08)",
            linecolor="rgba(28,35,51,0.15)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(28,35,51,0.08)",
            linecolor="rgba(28,35,51,0.15)",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
)

pio.templates["equity"] = PLOTLY_TEMPLATE
pio.templates.default = "equity"


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        h1 {
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #1c2333;
        }

        h2, h3, h4 {
            font-weight: 600;
            letter-spacing: -0.01em;
            color: #1c2333;
        }

        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
            border-right: 1px solid #dde2ea;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #5c6578;
        }

        [data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
        }

        div[data-testid="stTabs"] button {
            font-weight: 500;
            font-size: 0.875rem;
        }

        .app-header {
            margin-bottom: 0.25rem;
        }

        .app-tagline {
            color: #5c6578;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, tagline: str) -> None:
    st.markdown(f'<p class="app-header"></p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="app-tagline">{tagline}</p>', unsafe_allow_html=True)


def apply_chart_defaults(fig):
    fig.update_layout(template="equity")
    return fig

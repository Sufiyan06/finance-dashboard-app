import streamlit as st

from finance_dashboard.data.news import fetch_company_news
from finance_dashboard.models import SidebarSelection


def render_news_tab(selection: SidebarSelection) -> None:
    st.subheader("News and events")
    primary = selection.primary_ticker
    if primary is None:
        st.info("Select a focus ticker in the sidebar.")
        return

    company_name = selection.ticker_to_company.get(primary, primary)
    st.caption(f"Headlines for {company_name} ({primary})")

    news_days = st.sidebar.slider("News lookback (days)", 7, 30, 30)
    articles = fetch_company_news(company_name, days_back=news_days)

    if not articles:
        st.info("No recent articles found for this company.")
        return

    for article in articles:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        if article.get("urlToImage"):
            st.image(article["urlToImage"], width=400)
        st.caption(f"{article['source']['name']} · {article['publishedAt'][:10]}")
        st.write(article.get("description", ""))
        st.divider()

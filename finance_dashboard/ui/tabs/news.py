import streamlit as st

from finance_dashboard.config import get_news_api_key
from finance_dashboard.data.news import NewsResult, fetch_news_for_company
from finance_dashboard.models import SidebarSelection

_SESSION_KEY = "user_news_api_key"


def _render_articles(result: NewsResult) -> None:
    for article in result.articles:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        if article.get("urlToImage"):
            st.image(article["urlToImage"], width=400)
        st.caption(f"{article['source']['name']} · {article['publishedAt'][:10]}")
        st.write(article.get("description", ""))
        st.divider()


def _render_byo_key_guide() -> None:
    st.markdown(
        """
        **Quick setup (~2 minutes)**

        1. Open [newsapi.org/register](https://newsapi.org/register) and create a free account  
        2. After login, copy your **API key** from the dashboard  
        3. Paste it below and click **Load news**

        Your key stays in this browser session only — it is not stored on our servers.
        """
    )
    user_key = st.text_input("Your NewsAPI key", type="password", key="byo_news_api_key")
    if st.button("Load news with my key", type="primary", use_container_width=True):
        if not user_key.strip():
            st.warning("Paste your API key first.")
        else:
            st.session_state[_SESSION_KEY] = user_key.strip()
            st.rerun()


def render_news_tab(selection: SidebarSelection) -> None:
    st.subheader("News and events")
    primary = selection.primary_ticker
    if primary is None:
        st.info("Select a focus ticker in the sidebar.")
        return

    company_name = selection.ticker_to_company.get(primary, primary)
    st.caption(f"Headlines for {company_name} ({primary})")

    news_days = st.sidebar.slider("News lookback (days)", 7, 30, 30)
    user_key = st.session_state.get(_SESSION_KEY)
    using_own_key = bool(user_key)

    if using_own_key:
        st.caption("Using your NewsAPI key for this session.")
        if st.button("Switch back to shared demo key", use_container_width=True):
            del st.session_state[_SESSION_KEY]
            st.rerun()

    result = fetch_news_for_company(company_name, days_back=news_days, user_api_key=user_key)

    if result.status == "ok":
        _render_articles(result)
        return

    if result.status == "empty":
        st.info("No recent articles found for this company.")
        return

    if result.status == "no_key":
        st.warning("News is unavailable until a shared or personal API key is provided.")
        with st.expander("Add your free NewsAPI key", expanded=True):
            _render_byo_key_guide()
        return

    if result.status == "rate_limited":
        if using_own_key:
            st.warning("Your NewsAPI daily limit has been reached. Try again tomorrow.")
        else:
            st.warning(
                "The shared demo NewsAPI limit has been reached for today. "
                "All other tabs still work — try news again tomorrow, or use your own free key below."
            )
        with st.expander("Use your own free NewsAPI key", expanded=not using_own_key):
            _render_byo_key_guide()
        return

    if result.status == "invalid_key":
        st.error("That API key is not valid. Double-check it on [newsapi.org](https://newsapi.org/account).")
        if using_own_key:
            del st.session_state[_SESSION_KEY]
        with st.expander("Enter a valid NewsAPI key", expanded=True):
            _render_byo_key_guide()
        return

    st.error(f"Could not load news: {result.message or 'Unknown error'}")
    if not using_own_key and get_news_api_key():
        with st.expander("Try with your own NewsAPI key"):
            _render_byo_key_guide()

import streamlit as st

from gpt_app_learn.utils.page import YoutubeSummaryContent
from gpt_app_learn.utils.youtube import YouTubeRetriever


def main():
    page_content = YoutubeSummaryContent()
    page_content.init_page(
        page_title="Youtube Summarizer",
        page_icon="🤗",
        header_title="Youtube Summarizer 🤗",
    )
    page_content.init_costs()
    page_content.select_model()

    container = st.container()
    response_container = st.container()

    with container:
        url = YouTubeRetriever.get_url_input()
        youtube_retriever = YouTubeRetriever(url)
        if url:
            content = youtube_retriever.get_content()
            answer = page_content.prompt2answer(content)
        else:
            answer = None

    if answer:
        with response_container:
            page_content.print_summary(content, answer)

    page_content.print_costs()


if __name__ == "__main__":
    main()

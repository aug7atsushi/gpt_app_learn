import streamlit as st

from gpt_app_learn.utils.page import QAContent
from gpt_app_learn.utils.url import PageRetriever


def main():
    page_content = QAContent()
    page_content.init_page(
        page_title="Website Summarizer",
        page_icon="🤗",
        header_title="Website Summarizer 🤗",
    )
    page_content.select_model()
    page_content.init_messages()
    page_content.init_costs()

    container = st.container()
    response_container = st.container()

    answer = None
    with container:
        url = PageRetriever.get_url_input()
        page_retriever = PageRetriever(url)
        if page_retriever.is_validate_url():
            content = page_retriever.get_content()
            if content:
                answer = page_content.prompt2answer(content)
        else:
            st.write("Please input valid url")

    if answer:
        with response_container:
            page_content.print_summary(content, answer)

    page_content.print_costs()


if __name__ == "__main__":
    main()

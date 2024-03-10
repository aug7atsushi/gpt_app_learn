from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


class PageRetriever:
    def __init__(self, url: str) -> None:
        self.url = url

    @staticmethod
    def get_url_input() -> str:
        """ユーザから入力されたURLを取得"""
        url = st.text_input("URL: ", key="input")
        return url

    def is_validate_url(self) -> bool:
        """有効なURLかどうかを判定"""
        try:
            result = urlparse(self.url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def get_content(self) -> str:
        """URLを入力としてページのコンテンツを返す"""
        try:
            with st.spinner("Fetching Content ..."):
                response = requests.get(self.url)
                soup = BeautifulSoup(response.text, "html.parser")
                # fetch text from main (change the below code to filter page)
                if soup.main:
                    return soup.main.get_text()
                elif soup.article:
                    return soup.article.get_text()
                else:
                    return soup.body.get_text()
        except:
            st.write("something wrong")
            return None

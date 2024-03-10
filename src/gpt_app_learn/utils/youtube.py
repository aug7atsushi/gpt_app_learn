from urllib.parse import urlparse

import streamlit as st
from langchain_community.document_loaders import YoutubeLoader


class YouTubeRetriever:
    def __init__(self, url: str) -> None:
        self.url = url

    @staticmethod
    def get_url_input():
        """ユーザから入力されたURLを取得"""
        url = st.text_input("Youtube URL: ", key="input")
        return url

    def get_content(self):
        """URLを入力として`Document`の形式のデータを返す"""
        # NOTE: `Document`は`page_content`と`metadata`の2つのフィールドを持つ
        with st.spinner("Fetching Content ..."):
            loader = YoutubeLoader.from_youtube_url(
                self.url,
                add_video_info=True,  # タイトルや再生数も取得できる
                language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
            )
            return loader.load()


class LongYouTubeVideoRetriever(YouTubeRetriever):
    def __init__(self, url: str) -> None:
        super(LongYouTubeVideoRetriever, self).__init__(url)

    def get_content(self):
        """URLを入力として`Document`の形式のデータを返す"""
        # NOTE: `Document`は`page_content`と`metadata`の2つのフィールドを持つ
        with st.spinner("Fetching Content ..."):
            loader = YoutubeLoader.from_youtube_url(
                self.url,
                add_video_info=True,  # タイトルや再生数も取得できる
                language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
            )

        return loader.load_and_split()

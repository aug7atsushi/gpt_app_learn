import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


class PDFRetriever:
    def __init__(self) -> None:
        self.uploaded_file = st.file_uploader(
            label="Upload your PDF here😇", type="pdf"
        )

    def get_chunked_text(self):
        if self.uploaded_file:
            pdf_reader = PdfReader(self.uploaded_file)
            text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="text-embedding-ada-002",
                # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
                # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
                # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
                chunk_size=500,
                chunk_overlap=0,
            )
            return text_splitter.split_text(text)
        else:
            return None

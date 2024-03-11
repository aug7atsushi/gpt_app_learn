import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms.openai import OpenAI
from langchain_openai import ChatOpenAI

from gpt_app_learn.utils.pdf import PDFRetriever
from gpt_app_learn.utils.qdrant import QdrantVectorStore


class PageContentBase:
    def __init__(self) -> None:
        self.session_state = st.session_state

    def init_page(
        self,
        page_title: str,
        page_icon: str,
        header_title: str,
        sidebar_title="Options",
    ) -> None:
        """ページの初期化"""
        st.set_page_config(page_title, page_icon)
        st.header(header_title)
        st.sidebar.title(sidebar_title)

    def init_messages(
        self,
        button_msg="Clear Conversation",
        button_key="clear",
        system_default_role="You are a helpful assistant.",
    ) -> None:
        """チャット履歴の初期化"""
        clear_button = st.sidebar.button(button_msg, key=button_key)
        if clear_button or "messages" not in self.session_state:
            self.session_state.messages = [SystemMessage(content=system_default_role)]

    def init_costs(self):
        self.session_state.costs = []

    def select_model(self) -> None:
        """モデルの選択"""
        model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
        if model == "GPT-3.5":
            model_name = "gpt-3.5-turbo"
        else:
            model_name = "gpt-4"

        # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
        # 初期値は0.0、刻み幅は0.01とする
        temperature = st.sidebar.slider(
            "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01
        )
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

    def print_costs(self):
        costs = self.session_state.get("costs", [])
        st.sidebar.markdown("## Costs")
        st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            st.sidebar.markdown(f"- ${cost:.5f}")


class QAContent(PageContentBase):
    def __init__(self) -> None:
        super(QAContent, self).__init__()

    @staticmethod
    def get_prompt(content: str, n_chars: int = 300):
        return f"""以下はとあるWebページのコンテンツである。
        内容を{n_chars}字程度でわかりやすく要約してください。
        ========{content[:1000]}========
        日本語で書いてね！
        """

    def get_answer(self):
        print(self.session_state.messages)
        with get_openai_callback() as cb:
            answer = self.llm(self.session_state.messages)
        return answer.content, cb.total_cost

    def prompt2answer(self, content: str):
        # プロンプトを取得
        prompt = QAContent.get_prompt(content)
        self.session_state.messages.append(HumanMessage(content=prompt))

        # GPTの回答を取得
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = self.get_answer()
        self.session_state.costs.append(cost)
        # TODO:AIMessageも追加したい
        # self.session_state.messages.append(AIMessage)

        print(answer, cost)
        return answer

    def print_summary(self, content, answer):
        st.markdown("## Summary")
        st.write(answer)
        st.markdown("---")
        st.markdown("## Original Text")
        st.write(content)


class YoutubeSummaryContent(PageContentBase):
    def __init__(self) -> None:
        super(YoutubeSummaryContent, self).__init__()

    @staticmethod
    def get_prompt():
        return """Write a concise Japanese summary of the following transcript
        of Youtube Video.
        ============

        {text}

        ============

        ここから日本語で書いてね
        必ず3段落以内の200文字以内で簡潔にまとめること:
        """

    def get_prompt_template(self):
        prompt = YoutubeSummaryContent.get_prompt()
        self.prompt_template = PromptTemplate(template=prompt, input_variables=["text"])

    def get_answer(self, content):
        with get_openai_callback() as cb:
            # 要約の実行
            # 指定されたLLMを用いて、与えられた Documentのテキストの要約を行う
            chain = load_summarize_chain(
                self.llm, chain_type="stuff", verbose=True, prompt=self.prompt_template
            )
        response = chain({"input_documents": content}, return_only_outputs=True)
        return response["output_text"], cb.total_cost

    def prompt2answer(self, content: str):
        # プロンプトを取得
        self.get_prompt_template()

        # GPTの回答を取得
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = self.get_answer(content)
        self.session_state.costs.append(cost)
        print(answer, cost)
        return answer

    def print_summary(self, content, answer):
        st.markdown("## Summary")
        st.write(answer)
        st.markdown("---")
        st.markdown("## Original Text")
        st.write(content)


class LongYoutubeSummaryContent(YoutubeSummaryContent):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_prompt():
        return """Write a concise Japanese summary of the following transcript
        of Youtube Video.

        {text}

        ここから日本語で書いてね:
        """

    def get_answer(self, content):
        with get_openai_callback() as cb:
            # 要約の実行
            # 指定されたLLMを用いて、与えられた Documentのテキストの要約を行う
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                verbose=True,
                map_prompt=self.prompt_template,
                combine_prompt=self.prompt_template,
            )
        response = chain({"input_documents": content}, return_only_outputs=True)
        return response["output_text"], cb.total_cost


class PDFQAContent(PageContentBase):
    def __init__(self, qdrant_root: str, collection_name: str) -> None:
        super(PDFQAContent, self).__init__()
        self.qdrant_root = qdrant_root
        self.collection_name = collection_name

    def select_model(self, n_token_instruction: int = 300) -> None:
        """モデルの選択"""
        model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
        if model == "GPT-3.5":
            self.session_state.model_name = "gpt-3.5-turbo"
        elif model == "GPT-3.5":
            self.session_state.model_name = "gpt-3.5-turbo-16k"
        else:
            self.session_state.model_name = "gpt-4"

        # NOTE: 本文以外の指示のトークン数を差し引く
        self.session_state.max_token = (
            OpenAI.modelname_to_contextsize(self.session_state.model_name)
            - n_token_instruction
        )

        # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
        # 初期値は0.0、刻み幅は0.01とする
        temperature = st.sidebar.slider(
            "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01
        )
        self.llm = ChatOpenAI(
            temperature=temperature, model_name=self.session_state.model_name
        )

    def get_answer(self, qa, query):
        with get_openai_callback() as cb:
            # query / result / source_documents
            answer = qa(query)

        return answer, cb.total_cost

    def prompt2answer(self, vector_store):
        query = st.text_input("Query: ", key="input")

        answer = None
        if query:
            # QAモデルを取得
            qa = vector_store.get_qa_model(self.llm)
            # QAモデルの回答を取得
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = self.get_answer(qa, query)
                st.session_state.costs.append(cost)
            print(answer, cost)
        return answer

    def print_answer(self, answer):
        st.markdown("## Answer")
        st.write(answer)

    def switch_page(self) -> None:
        """ページの切り替え"""
        selection = st.sidebar.radio("Go to", ["PDF Upload", "Ask My PDF(s)"])
        if selection == "PDF Upload":
            self.page_pdf_upload_and_build_vector_db()
        elif selection == "Ask My PDF(s)":
            self.page_ask_my_pdf()

    def page_pdf_upload_and_build_vector_db(self):
        """PDFをアップロード&ベクトル保存を行うページ"""
        st.title("PDF Upload")
        container = st.container()
        with container:
            pdf_retriever = PDFRetriever()
            pdf_text = pdf_retriever.get_chunked_text()
            if pdf_text:
                with st.spinner("Loading PDF ..."):
                    vector_store = QdrantVectorStore(
                        qdrant_root=self.qdrant_root,
                        collection_name=self.collection_name,
                    )
                    vector_store.save_pdf_text2vector_store(pdf_text)

    def page_ask_my_pdf(self):
        """GPTに質問を投げるページ"""
        st.title("Ask My PDF(s)")
        self.select_model()

        container = st.container()
        response_container = st.container()

        with container:
            # ベクトルDBの準備
            vector_store = QdrantVectorStore(
                qdrant_root=self.qdrant_root, collection_name=self.collection_name
            )
            answer = self.prompt2answer(vector_store)
            if answer:
                with response_container:
                    self.print_answer(answer)

from typing import Tuple

import streamlit as st
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI


def init_page() -> None:
    """ページの初期化"""
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages() -> None:
    """チャット履歴の初期化"""
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model() -> ChatOpenAI:
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

    return ChatOpenAI(temperature=temperature, model_name=model_name)


def get_answer(llm: ChatOpenAI, messages: list) -> Tuple[str, float]:
    """モデルへチャットを送信し、返信メッセージと累計コストを返す"""
    with get_openai_callback() as cb:
        answer = llm(messages)
    # print(type(answer.content), type(cb.total_cost))
    return answer.content, cb.total_cost


# Streamlitは
def main():
    init_page()

    llm = select_model()
    init_messages()

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    # チャット履歴の表示
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # サイドバーに累計コストを表示
    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == "__main__":
    main()

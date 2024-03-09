from langchain.schema import AIMessage  # ChatGPTの返答
from langchain.schema import HumanMessage  # 人間の質問
from langchain.schema import SystemMessage  # システムメッセージ
from langchain_openai import ChatOpenAI

# from langchain_community.chat_models import ChatOpenAI


llm = ChatOpenAI()  # ChatGPT APIを呼んでくれる機能


# message = "Hi, ChatGPT!"  # あなたの質問をここに書く
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content=message),
# ]
# response = llm(messages)
# print(response)


# message = "Hi, ChatGPT!"  # あなたの質問をここに書く
# messages = [
#     SystemMessage(content="絶対に関西弁で返答してください"),
#     HumanMessage(content=message),
# ]
# response = llm(messages)
# print(response)

message = "ChatGPTとStreamlitでAIアプリを作る本を書く。タイトルを1個考えて。"
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=message),
]
for temperature in [0, 1, 2]:
    print(f"==== temp: {temperature}")
    llm = ChatOpenAI(temperature=temperature)
    for i in range(3):
        print(llm(messages).content)

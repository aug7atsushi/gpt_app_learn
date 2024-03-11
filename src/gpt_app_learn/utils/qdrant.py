from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantVectorStore:
    def __init__(
        self,
        qdrant_root: str,
        collection_name: str,
        vector_size: int = 1536,
        dist_metric=Distance.COSINE,
    ) -> None:
        self.client = QdrantClient(path=qdrant_root)
        self.collection_name = collection_name
        self.vector_config = VectorParams(size=vector_size, distance=dist_metric)

        self._create_collection()

    def _create_collection(self):
        # すべてのコレクション名を取得
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        # コレクションが存在しなければ作成
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_config,
            )
            print("collection created")

    def get_qdrant(self):
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=OpenAIEmbeddings(),
        )

    def save_pdf_text2vector_store(self, pdf_text):
        qdrant = self.get_qdrant()
        qdrant.add_texts(pdf_text)
        # 以下のようにもできる。この場合は毎回ベクトルDBが初期化される
        # LangChain の Document Loader を利用した場合は `from_documents` にする
        # Qdrant.from_texts(
        #     pdf_text,
        #     OpenAIEmbeddings(),
        #     path="./local_qdrant",
        #     collection_name="my_documents",
        # )

    def get_qa_model(self, llm):
        qdrant = self.get_qdrant()
        retriever = qdrant.as_retriever(
            search_type="similarity",  # "mmr", "similarity_score_threshold"等
            search_kwargs={"k": 10},  # 文書を何個取得するか (default: 4)
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
        )

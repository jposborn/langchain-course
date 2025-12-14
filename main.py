import os

import chromadb
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma

load_dotenv()


if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    vectorstore = Chroma(
    client=chroma_client,
    collection_name="docs",
    embedding_function=embeddings,
)

    # vectorstore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"], embedding=embeddings
    # )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)

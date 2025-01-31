import os
from pathlib import Path
from typing import Callable
from typing import List

import bs4
import chromadb
import pandas as pd
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document as LangchainDocument
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents_profiles import all_in_one_agent
from supported_countries import supported_countries
import streamlit as st

def init_rag_agent_from_profile(
    temperature=0.2,
    llm_model="gpt-4o-mini",
    embedding_model="text-embedding-ada-002",
    chunk_size=1000,
    agent_profile=all_in_one_agent,
) -> RunnableWithMessageHistory:
    
    llm = init_llm(temperature=temperature, llm_model=llm_model)
    

        
    vector_store_from_client = init_chroma_vector_store(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        collection_name=agent_profile["collection_name"],
        folders=agent_profile["folders"],
    )

    conversational_rag_chain = init_conversational_rag_chain(
        vector_store=vector_store_from_client,
        llm=llm,
        qa_prompt_generation_function=agent_profile["qa_prompt_generation_function"],
    )

    return conversational_rag_chain


def init_llm(
    temperature: float = 0.2,
    llm_model: str = "gpt-4o-mini",
) -> ChatOpenAI:
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    return llm


def init_conversational_rag_chain(
    vector_store: Chroma,
    llm: ChatOpenAI,
    qa_prompt_generation_function: Callable = all_in_one_agent["qa_prompt_generation_function"],
) -> RunnableWithMessageHistory:
    # Initialize Chroma Vector Store

    retriever = vector_store.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    ### Answer question ###
    qa_system_prompt = qa_prompt_generation_function() + """{context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def init_chroma_vector_store(
    embedding_model,
    chunk_size,
    collection_name=all_in_one_agent["collection_name"],
    folders=all_in_one_agent["folders"],
) -> Chroma:
    embeddings = OpenAIEmbeddings(chunk_size=chunk_size, model=embedding_model)

    if not len(folders) == 0:
        persistent_client = chromadb.PersistentClient(path="./data/chroma_db/chroma_langchain_db")
    else:
        persistent_client = chromadb.PersistentClient(path="./")
    collection = persistent_client.get_or_create_collection(collection_name)

    if collection.count() == 0:
        documents = aggregate_local_documents_and_urls(folders=folders)
        documents_splitted = split_documents(documents, chunk_size)

        texts, ids, metadatas = [], [], []
        for i, doc in enumerate(documents_splitted):
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(str(i))

        embeddings_values = embeddings.embed_documents(texts)
        max_batch_size = 5461
        for start in range(0, len(texts), max_batch_size):
            end = min(start + max_batch_size, len(texts))
            collection.add(
                documents=texts[start:end],
                embeddings=embeddings_values[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store_from_client


def aggregate_local_documents_and_urls(folders):
    documents = []
    for folder in folders:
        documents = aggregate_documents_in_folder(folder, documents)

    return documents


def aggregate_documents_in_folder(folder: Path, documents: List[Document]) -> List[Document]:
    # for country in country_list:# to be discussed with Loan
    documents = aggregate_pdfs_loop(documents, folder)
    documents = aggregate_urls_loop(documents, folder)
    documents = aggregate_csvs_loop(documents, folder)
    return documents


def aggregate_pdfs_loop(documents: List[Document], folder: Path) -> List[Document]:
    for file in folder.iterdir():
        if file.suffix != ".pdf":
            continue
        loader = PyPDFLoader(file)
        metadata = {"Type": "commun"}
        loaded_documents = loader.load()

        # Add metadata to each document
        for doc in loaded_documents:
            # If doc is a string, convert it into a Document object
            if isinstance(doc, str):
                doc = Document(page_content=doc)

            # Add the metadata to the document
            doc.metadata.update(metadata)  # Assumes 'metadata' is a dictionary

            # Append the document to the list
            documents.append(doc)
        documents.extend(loaded_documents)
    return documents


def aggregate_urls_loop(documents: List[Document], folder) -> List[Document]:
    url_file = [f for f in os.listdir(folder) if f.endswith("_urls.txt")]

    if len(url_file) == 0:
        return documents

    url_file = url_file[0]

    url_list = []
    with open(folder / url_file) as file:
        for line in file:
            if line.startswith("https://"):
                url_list.append(line)
    if url_list:
        bs4_strainer = bs4.SoupStrainer()
        loader = WebBaseLoader(
            web_paths=url_list,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        metadata = {"Type": "commun"}
        loaded_documents = loader.load()

        # Add metadata to each document
        for doc in loaded_documents:
            # If doc is a string, convert it into a Document object
            if isinstance(doc, str):
                doc = Document(page_content=doc)

            # Add the metadata to the document
            doc.metadata.update(metadata)  # Assumes 'metadata' is a dictionary

            # Append the document to the list
            documents.append(doc)
        documents.extend(loader.load())

    return documents


def aggregate_csvs_loop(documents: List[Document], folder: Path) -> List[Document]:
    for file in folder.iterdir():
        if file.suffix != ".csv":
            continue

        if file.name.startswith("recipes"):
            documents = recipes_csv_process(documents=documents, folder=folder)
        elif file.name == "production_norm_filtered.csv":
            documents = production_csv_process(documents=documents, folder=folder)

        elif file.name == "fs_norm_filtered.csv":
            documents = fs_csv_process(documents=documents, folder=folder)

    return documents


def recipes_csv_process(documents: List[Document], folder: Path) -> List[Document]:
    recipes_df = pd.read_csv(folder / "recipes_1.csv").dropna()
    if not recipes_df.empty:
        loader = DataFrameLoader(recipes_df, page_content_column="Sentence")
        documents.extend(loader.load())
    else:
        print(f"No recipes found.")
    return documents


def fs_csv_process(documents: List[Document], folder: Path) -> List[Document]:
    for country in supported_countries:
        fs_norm_filtered = pd.read_csv(folder / "fs_norm_filtered.csv")
        fs_norm_df_filtered_area = fs_norm_filtered[fs_norm_filtered["Area"] == country]
        fs_norm_df_filtered_area.fillna("Data not available", inplace=True)

        if not fs_norm_df_filtered_area.empty:
            loader = DataFrameLoader(fs_norm_df_filtered_area, page_content_column="Item")
            documents.extend(loader.load())
    return documents


def production_csv_process(documents: List[Document], folder: Path) -> List[Document]:
    production_norm_filtered_world = pd.read_csv(folder / "production_norm_filtered.csv")
    for country in supported_countries:
        if country is not None:
            production_norm_filtered = production_norm_filtered_world[
                production_norm_filtered_world["Area"] == country
            ][["Area", "Item", "Sentence"]]
        else:
            production_norm_filtered = production_norm_filtered[["Area", "Item", "Sentence"]]

        if not production_norm_filtered.empty:
            loader = DataFrameLoader(production_norm_filtered, page_content_column="Sentence")
            documents.extend(loader.load())
        else:
            print(f"No Production data found for the country {country}.")
    return documents


def split_documents(documents: List[LangchainDocument], chunk_size: int):
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)

    texts = []
    for doc in documents:
        texts += text_splitter.split_documents([doc])
    return texts

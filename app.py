import streamlit as st
import datetime
import os
from pathlib import Path
from typing import List
from uuid import uuid4

import pandas as pd
import bs4
import chromadb
import pycountry
import io 
from profiles import predefined_profiles

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document as LangchainDocument
from langchain.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DataFrameLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from data.country_list import country_list

# Initialize LLM and embeddings
def init_llm():
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=st.session_state.temperature)
    embeddings = OpenAIEmbeddings(chunk_size=st.session_state.chunk_size, model=st.session_state.embedding_model_name)
    init_chroma_vector_store(embeddings)

# Initialize Chroma Vector Store
def init_chroma_vector_store(embeddings):
    persistent_client = chromadb.PersistentClient(path="./data/chroma_db/chroma_langchain_db")
    collection = persistent_client.get_or_create_collection("cocineco_collection")

    if collection.count() == 0:
        documents = process_local_documents()
        documents_splitted = split_documents(documents, st.session_state.chunk_size)

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

    st.session_state.vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name="cocineco_collection",
        embedding_function=embeddings,
    )

# Document Processing Functions
def process_local_documents():
    documents = []
    documents = process_heatlh_risks_documents(documents)
    documents = process_diet_guidelines_documents(documents)
    documents = process_agriculture_documents(documents)
    documents = process_recipes_documents(documents)
    return documents

def split_documents(documents: List[LangchainDocument], chunk_size: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def process_heatlh_risks_documents(documents: List[Document]):
    for country in country_list:
        health_risks_folder = Path("data/health_risks")
        fs_norm_filtered = pd.read_csv(health_risks_folder / "fs_norm_filtered.csv")
        fs_norm_df_filtered_area = fs_norm_filtered[fs_norm_filtered["Area"] == country]
        fs_norm_df_filtered_area.fillna("Data not available", inplace=True)

        if not fs_norm_df_filtered_area.empty:
            loader = DataFrameLoader(fs_norm_df_filtered_area, page_content_column="Item")
            documents.extend(loader.load())

        documents = process_pdfs_loop(documents, health_risks_folder)
        documents = process_urls_loop(documents, health_risks_folder / "health_risks_urls.txt")

    return documents


def process_diet_guidelines_documents(documents: List[Document]) -> List[Document]:

    diet_guidelines_folder = Path("data/diet_guidelines")
    documents = process_pdfs_loop(documents=documents, folder=diet_guidelines_folder)
    documents = process_urls_loop(
        documents=documents, url_file=diet_guidelines_folder / "diet_guidelines_urls.txt"
    )

    return documents


def process_agriculture_documents(documents: List[Document]) -> List[Document]:
    for country in country_list:
        agriculture_folder = Path("data/agriculture")
        production_norm_filtered = pd.read_csv(agriculture_folder / "production_norm_filtered.csv")
        if country is not None:
            production_norm_filtered = production_norm_filtered[
                production_norm_filtered["Area"] == country
            ][["Area", "Item", "Sentence"]]
        else:
            production_norm_filtered = production_norm_filtered[["Area", "Item", "Sentence"]]

        if not production_norm_filtered.empty:
            loader = DataFrameLoader(production_norm_filtered, page_content_column="Sentence")
            documents.extend(loader.load())
        else:
            print(f"No Production data found for the country {country}.")

        documents = process_pdfs_loop(documents=documents, folder=agriculture_folder)
        documents = process_urls_loop(
            documents=documents, url_file=agriculture_folder / "agriculture_urls.txt"
        )
        if country is not None:

            fao_url = f"https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/{country.lower()}/en/"

            bs4_strainer = bs4.SoupStrainer()
            loader = WebBaseLoader(
                web_paths=[fao_url],
                bs_kwargs={"parse_only": bs4_strainer},
            )
            documents.extend(loader.load())
    return documents


def process_recipes_documents(documents: List[Document]) -> List[Document]:

    recipes_folder = Path("data/recipes")
    documents = process_pdfs_loop(documents=documents, folder=recipes_folder)
    documents = process_urls_loop(documents=documents, url_file=recipes_folder / "recipes_urls.txt")

    recipes_df = pd.read_csv(recipes_folder / "recipes_1.csv").dropna()
    if not recipes_df.empty:
        loader = DataFrameLoader(recipes_df, page_content_column="Sentence")
        documents.extend(loader.load())
    else:
        print(f"No recipes found.")

    return documents


def process_pdfs_loop(documents: List[Document], folder: Path) -> List[Document]:
    
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


def process_urls_loop(documents: List[Document], url_file: Path) -> List[Document]:
    url_list = []
    with open(url_file) as file:
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


def process_new_profile():

    reset_chat_history()
    # Split the document into chunks
    vector_store = st.session_state.vector_store_from_client

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

    history_aware_retriever = create_history_aware_retriever(st.session_state.llm, retriever, contextualize_q_prompt)
    today = datetime.date.today().strftime("%Y-%m-%d")
    ### Answer question ###
    qa_system_prompt = (
        "You are a Nutrition assistant for question-answering tasks."
        + f"The date is {today}. "
        + f"You are speaking to a  user of {st.session_state.gender} gender, of {st.session_state.age}years of age,"
        + f"with a size of {st.session_state.height}  cm and a weight of {st.session_state.weight}  kg from the country {st.session_state.country}."
        + "you need to help this person with their diet."
        + "Using the information contained in the context,"
        + "you will initially ask one after the other, 3 questions to the end user about their health"
        + "if someone doesn't answer one of your question, you will re-ask it up to 3 times."
        + "also ask them about their social habits like drinking or smoking and the frequency"
        "then you will ask them if they have particular allergies, intolerences or food preferences."
        + "After that, using the information contained in the context"
        + "you will identify 25 ingredients produced in the country of the user and available in this season"
        + " you will ask the user if these ingredients are ok for them to eat."
        + "After that, Using the information contained in the context,"
        + "you will create a 1 week meal plan with snacks in between meals  in a csv format between triple quote marks that is optimised for the user health and"
        + "that is based on the previous ingredients"
        + "the 1 week meal plan should contain the amount of calories, serving size, fats, carbohydrates, sugars, proteins, Percent Daily Value, calcium, iron, potassium, and fiber for each meal"
        + "mention the total of calories each day "
        + "you will not use expressions such as 'season vegetables' or 'season fruits' but instead you will use the names"
        + " of the fruits and vegetables to eat in this season and in this country"
        + "suggest calroie intake for the user based on their BMI in another response"
        + "also suggest some exercises to go with the meal plan in another response"
        + "also optimised to maximise the consumption of locally produced food and of seasonal products."
        + "You will then ask the user if their is something you should correct in this plan."
        + "If necessary you will correct this plan and re-submit it to the user."
        + "Finally you will produce a csv file containing the final meal plan."
        + "If you are asked questions about anything else but health, nutrition, agriculture, food or diet, you will answer that you don't know."
        + """If you don't know the answer, just say that you don't know. \

    {context}"""
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# Process User Prompt
def process_prompt(prompt):

    output = st.session_state.conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "abc123"}},
    )
    answer = output["answer"]

    st.session_state.chat_history.append((prompt, answer))
    return answer


def send_meal_plan(answer):
    # Generate a filename with the current timestamp
    
    file_name = f"Meal-Plan_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"

    # Extract the CSV content from the answer
    csv_content = answer.split("```")[1]

    # Prepare the CSV file content (you can customize this logic)
    # Convert the CSV string into a binary stream for download
    file_buffer = io.StringIO(csv_content)

    # Replace the placeholder message with a user-friendly message
    answer = answer.replace(
        answer.split("```")[1],
        "Get ready to nourish your body! Your customized Meal Plan, blending sustainability, nutrition, and your unique health needs is now downloadable.",
    )
    answer = answer.replace("```", "")

    # Show the answer text in the app
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    # Add a download button
    st.download_button(
        label="Download Meal Plan",
        data=file_buffer.getvalue(),
        file_name=file_name,
        mime="text/csv"
    )

def reset_chat_history():
    st.session_state.chat_history = []

profile_choice = st.sidebar.selectbox(
    "Select Profile", 
    list(predefined_profiles.keys()),
    on_change=process_new_profile)


# Retrieve the selected profile's data
profile_data = predefined_profiles[profile_choice]

# Auto-fill fields based on the selected profile
gender_profile = profile_data["gender"]
age_profile = profile_data["age"]
size_profile = profile_data["size"]
weight_profile = profile_data["weight"]
country_profile = profile_data["country"]

# Sidebar fields update dynamically based on selected profile

st.sidebar.selectbox(
    "Gender",
    ["male", "female"],
    index=["male", "female"].index(gender_profile),
    on_change=process_new_profile,
    key='gender'
)
st.sidebar.number_input(
    "Select Your Age",
    min_value=10,
    max_value=100,
    value=age_profile, 
    step=1,
    on_change=process_new_profile,
    key='age'
)
st.sidebar.number_input(
    "Select Your Height",
    min_value=120,
    max_value=210,
    value=size_profile, 
    step=1,
    on_change=process_new_profile,
    key='height'
)

st.sidebar.number_input(
    "Enter Your Weight (kg)",
    min_value=20,  # Minimum weight
    max_value=300,  # Maximum weight
    value=weight_profile,  # Default weight
    step=1,
    on_change=process_new_profile,
    key='weight'
)

# Get a list of all country names using pycountry
countries = [country.name for country in pycountry.countries]

st.sidebar.selectbox(
    "Select Your Country", 
    countries, 
    index=countries.index(country_profile),
    on_change=process_new_profile,
    key='country'
    )
    

st.title("CocinEco by A3I-Data Science")
st.sidebar.title("Settings")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "bot_initialized" not in st.session_state:
    st.session_state.bot_initialized = False

if not st.session_state.bot_initialized:
    # Global Variables stored in st.session_state
    st.session_state.llm = None
    st.session_state.embeddings = None
    st.session_state.chunk_size = 1000
    st.session_state.embedding_model_name = "text-embedding-3-small"
    st.session_state.temperature = 0.2
    st.session_state.vector_store_from_client = None
    st.session_state.conversational_rag_chain = None
    st.session_state.chat_history = []
    
    chat_message = "Hello there! I'm CocinEco: an AI assistant that will help you elaborate sustainable meal plans. I will ask you a few questions to understand you better and provide you with personalized nutrition advice. Let's get started! ok?"

    # Initialize chatbot
    st.chat_message("assistant").write(chat_message)
    st.session_state.messages.append({"role": "assistant", "content": chat_message})

    # Initialize LLM and Profile
    init_llm()
    process_new_profile()
    
    st.session_state.bot_initialized = True



if prompt := st.chat_input():
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(prompt)

    with st.spinner("Processing..."):
        answer = process_prompt(prompt)
    if "```" in answer:
        send_meal_plan(answer)
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)



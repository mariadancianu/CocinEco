import datetime
import io
import logging
import logging.config
import os
import sqlite3

import bcrypt
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory

from agents_profiles import all_in_one_agent
from Main_Menu_app import init_cocineco
from RAG_agent_definition import init_rag_agent_from_profile
from supported_countries import supported_countries
from user_profiles import predefined_profiles

load_dotenv()  # Load environment variables

logger = logging.getLogger("app")


debug_mode = os.getenv("DEBUG", False)
logger.debug(f"debug mode: {debug_mode}")

logging.basicConfig(
    filename="LogFile.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# Process User Prompt
def process_prompt(prompt, conversational_rag_chain: RunnableWithMessageHistory, chat_history):
    output = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "abc123"}},
    )
    answer = output["answer"]
    context = output["context"]

    chat_history.append((prompt, answer))
    return answer, context, chat_history


def save_meal_plan_to_csv(answer, file_name):
    if st.session_state.user_name != "Unnamed User":
        with open(file_name, "w") as text_file:
            text_file.write(answer.split("```")[1])


def send_meal_plan(answer):
    # Generate a filename with the current timestamp

    if st.session_state.user_name != "Unnamed User":
        file_name = f"Meal-Plan_{st.session_state.user_name}_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
    else:
        file_name = f"Meal-Plan_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"

    # Extract the CSV content from the answer
    csv_content = answer.split("```")[1]

    save_meal_plan_to_csv(answer, file_name)

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
        mime="text/csv",
    )


def update_fields_with_profile(profile_data):

    st.session_state.gender = profile_data["gender"]
    st.session_state.age = profile_data["age"]
    st.session_state.height = profile_data["height"]
    st.session_state.weight = profile_data["weight"]
    st.session_state.country = profile_data["country"]
    st.session_state.health_condition_list = profile_data["health_condition_list"]
    st.session_state.other_health_condition_str = profile_data["other_health_condition_str"]
    st.session_state.allergies_list = profile_data["allergies_list"]
    st.session_state.other_allergies_str = profile_data["other_allergies_str"]
    st.session_state.food_intolerance_list = profile_data["food_intolerance_list"]
    st.session_state.other_food_intolerance_str = profile_data["other_food_intolerance_str"]

    if "user_name" in profile_data.keys():
        st.session_state.user_name = profile_data["user_name"]
    else:
        st.session_state.user_name = "Unnamed User"


def reset_rag_agent(
    temperature,
    llm_model,
    embedding_model,
    chunk_size,
    agent_profile,
):
    st.session_state.chat_history = []

    st.session_state.conversational_rag_chain = init_rag_agent_from_profile(
        temperature=temperature,
        llm_model=llm_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        agent_profile=agent_profile,
    )


def cocineco_is_ready_to_start():

    requiered_fields = ["gender", "age", "height", "weight", "country"]
    for fiel in requiered_fields:
        if fiel not in st.session_state:

            return False
        elif st.session_state[fiel] == None:
            return False
    return True


def initialize_session_state():

    if "bot_initialized" not in st.session_state:
        st.session_state.bot_initialized = False
    if "cocineco_has_started" not in st.session_state:
        st.session_state.cocineco_has_started = False
    if "restart_cocineco" not in st.session_state:
        st.session_state.restart_cocineco = False
    if "previous_profile_messages" not in st.session_state:
        st.session_state.previous_profile_messages = []

    if not st.session_state.bot_initialized:
        st.session_state.user_name = "Unnamed User"
        st.session_state.llm = None
        st.session_state.embeddings = None
        st.session_state.chunk_size = 1000
        st.session_state.embedding_model_name = "text-embedding-3-small"
        st.session_state.temperature = 0.2
        st.session_state.vector_store_from_client = None
        st.session_state.conversational_rag_chain = None
        st.session_state.chat_history = []
        st.session_state.agent_profile = all_in_one_agent


def show_profile_options():

    st.sidebar.selectbox(
        "Select Profile",
        list(predefined_profiles.keys()),
        on_change=lambda: update_fields_with_profile(
            predefined_profiles[st.session_state.profile_choice]
        ),
        key="profile_choice",
    )
    reset_rag_agent_args = (
        st.session_state.temperature,
        "gpt-4o-mini",
        st.session_state.embedding_model_name,
        st.session_state.chunk_size,
        st.session_state.agent_profile,
    )
    st.sidebar.segmented_control(
        "Gender",
        ["male", "female"],
        key="gender",
    )
    st.sidebar.slider(
        "Select Your Age",
        min_value=10,
        max_value=100,
        step=1,
        value=40,
        key="age",
    )
    st.sidebar.number_input(
        "Select Your Height (cm)",
        min_value=120,
        max_value=250,
        step=1,
        value=170,
        key="height",
    )

    st.sidebar.number_input(
        "Enter Your Weight (kg)",
        min_value=20,  # Minimum weight
        max_value=300,  # Maximum weight
        step=1,
        value=70,
        key="weight",
    )
    st.sidebar.segmented_control(
        "Select Your Country",
        supported_countries,
        key="country",
    )
    common_nutrition_health_conditions = [
        "Type 2 Diabetes",
        "Cardiovascular Disease",
        "Hypertension",
        "Osteoporosis",
        "Iron Deficiency Anemia",
    ]

    st.sidebar.segmented_control(
        "Health condition",
        common_nutrition_health_conditions,
        key="health_condition_list",
        selection_mode="multi",
    )
    st.sidebar.text_input(
        "Other Health Condition",
        key="other_health_condition_str",
        value="",
    )

    common_food_allergies = ["Peanuts", "Tree Nuts", "Milk", "Eggs", "Shellfish ", "Wheat"]

    st.sidebar.segmented_control(
        "Allergies",
        common_food_allergies,
        key="allergies_list",
        selection_mode="multi",
    )
    st.sidebar.text_input(
        "Other Allergies",
        key="other_allergies_str",
        value="",
    )

    common_food_intolerances = [
        "Lactose Intolerance",
        "Gluten Sensitivity (Non-Celiac)",
        "Fructose Malabsorption",
        "Histamine Intolerance",
        "FODMAP Intolerance",
        "Caffeine Sensitivity",
    ]

    st.sidebar.segmented_control(
        "Food Intolerance",
        common_food_intolerances,
        key="food_intolerance_list",
        selection_mode="multi",
    )
    st.sidebar.text_input(
        "Other Food Intolerance",
        key="other_food_intolerance_str",
        value="",
    )


def initialize_chatbot():
    st.session_state.chat_history = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if not st.session_state.bot_initialized or st.session_state.restart_cocineco:
        # Global Variables stored in st.session_state
        if "user_name" not in st.session_state:
            st.session_state.user_name = "UNAMED USER"

        chat_message = (
            f"Hello {st.session_state.user_name}! I'm CocinEco: an AI assistant that will help you elaborate sustainable meal plans."
            + "I will ask you a few questions to understand you better and provide"
            + "you with personalized nutrition advice. Let's get started! ok?"
        )

        # Initialize chatbot
        st.chat_message("assistant").write(chat_message)
        st.session_state.messages.append({"role": "assistant", "content": chat_message})

        reset_rag_agent(
            temperature=st.session_state.temperature,
            llm_model="gpt-4o-mini",
            embedding_model=st.session_state.embedding_model_name,
            chunk_size=st.session_state.chunk_size,
            agent_profile=all_in_one_agent,
        )

        st.session_state.bot_initialized = True


def run_conversation():

    number_of_messages = 0
    number_of_messages_max = 25

    if prompt := st.chat_input():
        number_of_messages = max(number_of_messages, len(st.session_state.messages))

        if number_of_messages < 20:
            with st.chat_message("user"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.write(prompt)
                logger.info("User : %s", st.session_state.user_name)
                logger.info(prompt)

            with st.spinner("Processing..."):
                answer, context, st.session_state.chat_history = process_prompt(
                    prompt,
                    conversational_rag_chain=st.session_state.conversational_rag_chain,
                    chat_history=st.session_state.chat_history,
                )
            if "```" in answer:
                number_of_messages = number_of_messages_max
                send_meal_plan(answer)
                logger.info("Assistant : %s", answer)
                logger.info("Context : %s", context)

                chat_message = "I hope you will find this plan useful ! For my part I am getting tired and will go to sleep until further notice"

                st.chat_message("assistant").write(chat_message)
                st.session_state.messages.append({"role": "assistant", "content": chat_message})

            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)
                if debug_mode:
                    context_str = "Context:\n"
                    for i, document in enumerate(context):
                        context_str += f"Document {i}:\nMetadata:\n{str(document.metadata)}\n"
                        context_str += f"Content:\n{str(document.page_content)}\n\n"

                    st.chat_message("assistant").write(context_str)
                    st.session_state.messages.append({"role": "assistant", "content": context_str})
                logger.info("Assistant : %s", answer)
                logger.info("Context : %s", context)
        else:
            chat_message = "Sorry, I have reached the maximum amount of work I can do in one day. I will go to sleep until further notice!"

            st.chat_message("assistant").write(chat_message)
            st.session_state.messages.append({"role": "assistant", "content": chat_message})


def main():

    init_cocineco()

    # Streamlit app layout

    if not st.session_state.authenfied_user:
        st.error("You must log in to view this page.")

    else:
        st.subheader("CocinEcoBot")

        initialize_session_state()

        if st.session_state.cocineco_has_started:
            if st.button("Restart CocinEcoBot"):
                if cocineco_is_ready_to_start():
                    st.session_state.restart_cocineco = True

                else:
                    st.warning(
                        "CocinEcoBot is not ready to start. Please fill the requiered fields."
                    )
        else:
            if st.button("Start CocinEcoBot"):
                if cocineco_is_ready_to_start():
                    st.session_state.restart_cocineco = True
                    st.session_state.cocineco_has_started = True

                else:
                    st.warning(
                        "CocinEcoBot is not ready to start. Please fill the requiered fields."
                    )

        st.write(
            "st.session_state.cocineco_has_started, st.session_state.restart_cocineco, cocineco_is_ready_to_start()"
        )
        st.write(
            st.session_state.cocineco_has_started,
            st.session_state.restart_cocineco,
            cocineco_is_ready_to_start(),
        )
        if cocineco_is_ready_to_start():
            st.write(st.session_state.gender, st.session_state.age, st.session_state.country)
        show_profile_options()

        if st.session_state.cocineco_has_started and cocineco_is_ready_to_start():
            # if st.session_state.restart_cocineco:
            initialize_chatbot()
            st.session_state.restart_cocineco = False
            run_conversation()


if __name__ == "__main__":
    main()

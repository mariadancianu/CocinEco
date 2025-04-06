import datetime
import io
import logging
import logging.config
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory

from agents_profiles import all_in_one_agent
from agents_profiles import all_in_one_agent_Chat_GPT
from Main_Menu import init_cocineco
from RAG_agent_definition import init_rag_agent_from_profile
from supported_countries import supported_countries
from user_profiles import build_predefined_user_system_prompt
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
st.set_page_config(
    page_title="CocinEco Individual Planner",
    page_icon="cocineco_browser_icon.png",
)


def save_meal_plan_to_csv(answer, file_name):
    if st.session_state.user_name != "Unnamed User":
        with open(file_name, "w") as text_file:
            text_file.write(answer.split("```")[1])


def send_meal_plan(answer):
    # Generate a filename with the current timestamp

    if st.session_state.user_name != "Unnamed User":
        st.session_state.file_name = f"Meal-Plan_{st.session_state.user_name}_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv".replace(
            " ", "_"
        )
    else:
        st.session_state.file_name = (
            f"Meal-Plan_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv".replace(
                " ", "_"
            )
        )

    # Extract the CSV content from the answer
    st.session_state.csv_content = answer.split("```")[1]

    save_meal_plan_to_csv(answer, st.session_state.file_name)

    # Prepare the CSV file content (you can customize this logic)
    # Convert the CSV string into a binary stream for download
    st.session_state.file_buffer = io.StringIO(st.session_state.csv_content)

    # Replace the placeholder message with a user-friendly message


def get_shopping_shopping_list_from_df(meal_plan_df):
    # @todo

    return meal_plan_df.Ingredients


def show_output_buttons():
    # Add a download button

    if st.session_state.file_name is not None:
        meal_plan_df = pd.read_csv(st.session_state.file_name, on_bad_lines="skip")

        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:

            # st.download_button(
            # label="Download displayed data as a CSV",
            # data=pd.read_csv(st.session_state.file_name),
            # file_name=st.session_state.file_name)

            st.download_button(
                label="Download Meal Plan",
                data=st.session_state.file_buffer.getvalue(),
                file_name=st.session_state.file_name,
                mime="text/csv",
            )
        with col2:
            if st.button("Hide/Show Meal Plan"):
                st.session_state["show_meal_plan"] = not st.session_state["show_meal_plan"]

        with col3:
            if st.button("Hide/Show Shopping List"):
                st.session_state["show_shopping_list"] = not st.session_state["show_shopping_list"]
                st.warning("Work In Progress!")

        if st.session_state.show_meal_plan:

            st.write(meal_plan_df)


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


def update_fields_with_profile(profile_data):

    st.session_state.gender = profile_data["gender"]
    st.session_state.age = profile_data["age"]
    st.session_state.height = profile_data["height"]
    st.session_state.weight = profile_data["weight"]
    st.session_state.country = profile_data["country"]
    st.session_state.health_conditions = profile_data["health_conditions"]

    st.session_state.allergies = profile_data["allergies"]

    st.session_state.food_intolerances = profile_data["food_intolerances"]

    if "user_name" in profile_data.keys():
        st.session_state.user_name = profile_data["user_name"]
    else:
        st.session_state.user_name = "Unnamed User"


def sidbar_inputs():

    st.sidebar.header("User Inputs")

    user_name = st.sidebar.selectbox(
        "User Name", ["Other"] + list(predefined_profiles.keys()), index=0
    )

    if user_name == "Other":
        st.session_state.user_name = st.sidebar.text_input("Enter Your Name")

        st.sidebar.segmented_control(
            "Gender",
            ["male", "female"],
            key="gender",
        )
        st.sidebar.number_input(
            "Age",
            min_value=10,
            max_value=100,
            step=1,
            value=None,
            key="age",
        )
        st.sidebar.number_input(
            "Height (cm)",
            min_value=120,
            max_value=250,
            step=1,
            value=None,
            key="height",
        )
        st.sidebar.number_input(
            "Weight (kg)",
            min_value=20,  # Minimum weight
            max_value=300,  # Maximum weight
            step=1,
            value=None,
            key="weight",
        )

        st.sidebar.segmented_control(
            "Select Your Country",
            supported_countries,
            key="country",
        )

        common_health_conditions = [
            "Type 2 Diabetes",
            "Cardiovascular Disease",
            "Hypertension",
            "Osteoporosis",
            "Iron Deficiency Anemia",
        ]

        common_food_intolerances = [
            "Lactose Intolerance",
            "Gluten Sensitivity (Non-Celiac)",
            "Fructose Malabsorption",
            "Histamine Intolerance",
            "FODMAP Intolerance",
            "Caffeine Sensitivity",
        ]

        common_food_allergies = ["Peanuts", "Tree Nuts", "Milk", "Eggs", "Shellfish ", "Wheat"]

        conditions_dict = {
            "allergies": common_food_allergies,
            "food_intolerances": common_food_intolerances,
            "health_conditions": common_health_conditions,
        }

        for cond_type in conditions_dict.keys():

            conditions = st.sidebar.segmented_control(
                cond_type.replace("_", " ").title(),
                conditions_dict[cond_type] + ["Other"],
                selection_mode="multi",
            )

            st.session_state[cond_type] = " ".join(conditions)
            if "Other" in conditions:
                st.session_state[cond_type] = st.session_state[cond_type].replace("Other", "")
                other_health_conditions = st.sidebar.text_input(
                    "Other " + cond_type.replace("_", " ").title(),
                    value="",
                )
                st.session_state[cond_type] = (
                    st.session_state[cond_type] + " and also " + other_health_conditions
                )

        reason_to_chat_with_cocineco = [
            "General health and wellness",
            "Weight management",
            "Managing chronic diseases",
            "Improving digestion and gut health",
            "Food allergies, intolerances, or special dietary needs",
            "Sports and performance",
        ]
        st.session_state.reason_to_chat_with_cocineco = st.sidebar.selectbox(
            "Reason to chat with Cocineco", reason_to_chat_with_cocineco, index=0
        )

    else:
        update_fields_with_profile(predefined_profiles[user_name])
        st.sidebar.success(
            build_predefined_user_system_prompt(predefined_profiles[user_name]).split("****")[0]
        )
        st.sidebar.success(
            build_predefined_user_system_prompt(predefined_profiles[user_name]).split("****")[1]
        )
        st.sidebar.success(
            build_predefined_user_system_prompt(predefined_profiles[user_name]).split("****")[2]
        )


def initialize_session_state():

    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # Stores the chat history
    if "bot_started" not in st.session_state:
        st.session_state["bot_started"] = False  # Bot running state
    if "show_user_info" not in st.session_state:
        st.session_state["show_user_info"] = True  # Toggle user info visibility
    if st.session_state.running_in_cloud:
        st.session_state.agent_profile = all_in_one_agent_Chat_GPT
    else:
        st.session_state.agent_profile = all_in_one_agent

    if not st.session_state.bot_started:
        st.session_state.llm = None
        st.session_state.embeddings = None
        st.session_state.chunk_size = 1000
        st.session_state.embedding_model_name = "text-embedding-3-small"
        st.session_state.temperature = 0.2
        st.session_state.vector_store_from_client = None
        st.session_state.conversational_rag_chain = None
        st.session_state.chat_history = []
        st.session_state.show_shopping_list = False

        st.session_state.file_name = None
        st.session_state.csv_content = None
        st.session_state.file_buffer = None
        st.session_state.show_meal_plan = False


def cocineco_is_ready_to_start():

    requiered_fields = ["gender", "age", "height", "weight", "country"]
    for fiel in requiered_fields:
        if fiel not in st.session_state:

            return False
        elif st.session_state[fiel] == None:
            return False
    return True


def initialize_cocineco_bot():

    st.session_state["bot_started"] = True
    if "user_name" not in st.session_state:
        st.session_state.user_name = "UNAMED USER"

    chat_message = (
        f"Hello {st.session_state.user_name}! I'm CocinEco: an AI assistant that will help you elaborate sustainable meal plans."
        + "I will ask you a few questions to understand you better and provide"
        + "you with personalized nutrition advice. Let's get started! ok?"
    )

    st.session_state.messages.append({"role": "assistant", "content": chat_message})

    reset_rag_agent(
        temperature=st.session_state.temperature,
        llm_model="gpt-4o-mini",
        embedding_model=st.session_state.embedding_model_name,
        chunk_size=st.session_state.chunk_size,
        agent_profile=st.session_state.agent_profile,
    )


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
                number_of_messages = number_of_messages_max - 5
                send_meal_plan(answer)
                logger.info("Assistant : %s", answer)
                logger.info("Context : %s", context)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # answer = "Your customized Meal Plan, blending sustainability, nutrition, and your  health needs is now available under the links below."
                # Show the answer text in the app
                # st.session_state.messages.append({"role": "assistant", "content": answer.split("```")[0] + answer.split("```")[2]})
                st.chat_message("assistant").write(answer.split("```")[0] + answer.split("```")[2])

                answer = "Is there anything you would like me to correct in this plan?"
                # Show the answer text in the app
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)

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


def show_missing_fields_message():

    st.warning("Please fill in all the requiered fields in the sidebar before restarting the bot.")
    for field in ["gender", "age", "height", "weight", "country"]:
        if st.session_state[field] == None:
            st.error(f"{field} is  {st.session_state[field]}")


def main():

    st.image("cocineco_banner_with_logo.png")

    init_cocineco()

    # Streamlit app layout

    if not st.session_state.authenticated_user:
        st.error("You must log in to view this page.")

    else:

        # Initialize session states
        initialize_session_state()

        # Sidebar for user inputs

        # Toggle user information visibility
        if st.sidebar.button("Hide/Show User Information"):
            st.session_state["show_user_info"] = not st.session_state["show_user_info"]

        if st.session_state["show_user_info"]:
            sidbar_inputs()
        # Function to generate the initial prompt

        # Main chat interface
        st.title("Individual Planner")

        # Buttons to control the bot
        col1, col2 = st.columns([1, 1])
        with col1:
            start_bot = st.button("Start Bot")
        with col2:
            restart_bot = st.button("Restart Bot")

        # Handle "Start Bot" button
        if start_bot:
            if cocineco_is_ready_to_start():
                initialize_cocineco_bot()
            else:
                show_missing_fields_message()

        # Handle "Restart Bot" button
        if restart_bot:
            if cocineco_is_ready_to_start():
                confirm_restart = st.radio(
                    "Are you sure you want to restart the chat?",
                    options=["No", "Yes"],
                    index=0,
                    key="confirm_restart",
                )
                if confirm_restart == "Yes":
                    st.session_state["bot_started"] = True

                    st.session_state["messages"].append(
                        {"role": "bot", "content": "Restarting bot..."}
                    )
                    st.session_state["bot_started"] = True
                    initialize_cocineco_bot()
            else:
                show_missing_fields_message()

        # Display the chat history
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

        # Chat input field (only active when bot has started)
        if st.session_state["bot_started"]:
            run_conversation()

        show_output_buttons()


if __name__ == "__main__":
    main()

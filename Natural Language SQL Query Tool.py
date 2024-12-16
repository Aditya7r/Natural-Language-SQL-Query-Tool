# Import required libraries
import boto3
import os
import tempfile
import streamlit as st
from typing import Dict, Any
import mysql.connector
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Bedrock
from botocore.exceptions import ClientError
import warnings
import whisper
import certifi
import ssl
import re
import pandas as pd
from langchain_community.chat_models import BedrockChat
from streamlit_js_eval import streamlit_js_eval

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Bedrock.*")

# Set default SSL context with certifi certificate
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# AWS credentials setup
os.environ['AWS_ACCESS_KEY_ID'] = 'AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'AWS_SECRET_ACCESS_KEY'

# LangChain setup for SQL Prompt Template
sql_prompt_template = """
You are a data analyst at a retail company. You are given a database of merchandise sales records, including order details, product categories, buyer demographics, pricing information, shipping details, and customer reviews. 

Based on the table schema below, write a MySQL query that would answer the user's question.

<SCHEMA>{schema}</SCHEMA>

Conversation History: {chat_history}

Question: {question}

Write only the SQL query, and do not include any additional text.
SQL Query:
"""

response_prompt_template = """
Based on the table schema, question, SQL query, and SQL response, provide a  natural language response. 

{schema}

Question: {question}

SQL Query: {query}

SQL Response: {response}

Please ensure that the response is structured clearly, highlighting key findings or records as requested in the user's question. Be very concise
"""

sql_prompt = ChatPromptTemplate.from_template(sql_prompt_template)
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# Database URI and setup
db_uri = 'mysql+mysqlconnector://root:your_password@localhost:3306/Merch_Sales'
db = SQLDatabase.from_uri(db_uri)


def get_schema1(_):
    return db._execute(
        "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE()"
    )


# Configure AWS Session for Bedrock
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-1'
)
bedrock_runtime_client = session.client("bedrock-runtime", region_name='us-east-1')

# Bedrock LLM setup
llm = Bedrock(
    model_id='amazon.titan-text-express-v1',
    client=bedrock_runtime_client,
    model_kwargs={"temperature": 0},
)


# MySQL Query Execution
def execute_mysql_query(query: str):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="your_password",
            database="Merch_Sales"
        )
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        connection.commit()
        cursor.close()
        connection.close()
        return result
    except mysql.connector.Error as e:
        return f"Error executing MySQL query: {str(e)}"


# Safe SQL execution
def safe_sql_execution(vars: Dict[str, Any]) -> str:
    query = vars["query"]
    try:
        result = execute_mysql_query(query)
        return result
    except Exception as e:
        return f"Error executing SQL: {str(e)}"


# Polly TTS function
def polly_tts(text):
    client = boto3.client('polly', region_name='us-east-1')
    try:
        response = client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as file:
            file.write(response['AudioStream'].read())
            return file.name
    except ClientError as e:
        return f"Error using Polly: {str(e)}"


# Log query, generated SQL, and response to database
def log_query_to_db(question: str, query: str, answer: str):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="your_password",
            database="Merch_Sales"
        )
        cursor = connection.cursor()
        insert_query = """
            INSERT INTO query_logs (input_question, query, output_answer)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (question, query, answer))
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as e:
        st.error(f"Error logging query: {str(e)}")


# Export chat history as CSV
def export_chat_history_to_csv(chat_history):
    if not chat_history:
        st.warning("No chat history to export!")
        return None

    # Convert chat history to a DataFrame
    df = pd.DataFrame(chat_history)
    csv_data = df.to_csv(index=False)  # Convert to CSV format
    return csv_data


# SQL Chain to generate query
sql_chain = (
        RunnablePassthrough.assign(schema=get_schema1)
        | sql_prompt
        | llm
        | StrOutputParser()
)

# Full Chain to generate response
full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema1,
            response=safe_sql_execution
        )
        | response_prompt
        | llm
        | StrOutputParser()
)


# Cache the Whisper model to avoid repeated downloads
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


whisper_model = load_whisper_model()


# Export chat history as CSV
# Export chat history as CSV
def export_chat_history_to_csv(chat_history):
    # Check if chat history exists
    if not chat_history:
        st.warning("No chat history to export!")
        return None

    # Convert chat history to a DataFrame
    df = pd.DataFrame(chat_history)
    csv_data = df.to_csv(index=False)  # Convert to CSV format
    return csv_data


# Streamlit App
def main():
    st.markdown("""
        <style>
            .stButton button {
                background-color: rgba(1, 0, 128, 255);
                color: white;
                border-radius: 5px;
                font-size: 16px;
                padding: 8px 16px;
                border: none;
            }
            .stButton button:hover {
                background-color: rgba(0, 0, 102, 255);
            }
        </style>
    """, unsafe_allow_html=True)



    st.title("Natural Language SQL Query Tool")
    developer_mode = st.sidebar.checkbox("Developer Options (Show SQL Query)", value=False)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Add audio input to sidebar
    audio_value = st.sidebar.audio_input("Record a voice message")

    # Get inputs
    user_question = st.chat_input("Ask your database query:")

    # Process audio input if present
    if audio_value:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_value.getvalue())
            temp_audio_file.flush()
            audio_text = whisper_model.transcribe(temp_audio_file.name)["text"]
            st.sidebar.write(f"Transcribed audio: {audio_text}")
            user_question = user_question or audio_text  # Use audio if no text is provided

        # Clear Chat History button
    if st.sidebar.button("Clear Chat History"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    if user_question:
        chat_history = "\n".join(
            [f"Question: {msg['question']}\nSQL Query: {msg['query']}" for msg in st.session_state.chat_history])

        # Generate SQL and response
        generated_sql = sql_chain.invoke({'question': user_question, 'chat_history': chat_history})
        final_response = full_chain.invoke({
            'question': user_question,
            'query': generated_sql,
            'schema': get_schema1(None),
            'chat_history': chat_history
        })

        # Log the query, SQL, and response
        log_query_to_db(user_question, generated_sql, final_response)

        # Append question and response to chat history
        st.session_state.chat_history.append(
            {"question": user_question, "query": generated_sql, "response": final_response}
        )

        # Display conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(f"**Q:** {msg['question']}")
            with st.chat_message("assistant"):
                st.write(f"**A:** {msg['response']}")

        

        # Format response for TTS
        formatted_response = final_response  # re.sub(r'[^\w\s]', '', final_response)
        audio_path = polly_tts(formatted_response)
        if audio_path:
            st.audio(audio_path, format="audio/mp3")

        # Developer mode: show SQL query
        if developer_mode:
            st.subheader("Generated SQL Query")
            st.code(generated_sql, language='sql')

        # Initialize chat history in session state if not already done
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Add export button to the sidebar
        if st.session_state.chat_history:
            csv_data = export_chat_history_to_csv(st.session_state.chat_history)
            st.sidebar.download_button(
                label="Export Chat History",
                data=csv_data,
                file_name="chat_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

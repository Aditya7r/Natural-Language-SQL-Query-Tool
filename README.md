# 🛠️ Natural-Language-SQL-Query-Tool

This application enables users to query MySQL databases using natural language through an intuitive **Streamlit** interface. Leveraging AWS Bedrock LLMs, this tool transforms user queries into executable SQL and provides responses in natural language, including optional **text-to-speech (TTS)** output using Amazon Polly.

---

## 🚀 Key Features

- **🗣️ Natural Language to SQL**: Converts user queries into SQL using AWS Bedrock LLMs.
- **🎤 Voice Input Support**: Transcribe audio queries using **OpenAI's Whisper**.
- **💾 Dynamic MySQL Query Execution**: Executes generated SQL queries on a MySQL database.
- **📝 Natural Language Responses**: Returns SQL query results as concise, natural language answers.
- **🔊 TTS Output**: Generates audio responses using **Amazon Polly**.
- **📥 Chat History Export**: Save chat history as a CSV file for easy record-keeping.
- **🛠️ Developer Mode**: Display generated SQL queries for debugging or learning purposes.
- **🔒 Secure AWS Integration**: Configurable AWS credentials for Bedrock and Polly services.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: MySQL, AWS Bedrock, Amazon Polly
- **Libraries**:
  - `boto3` for AWS services
  - `whisper` for speech-to-text
  - `langchain` for LLM-powered SQL generation
  - `streamlit_js_eval` for enhanced Streamlit functionality

---

## ⚙️ How It Works

1. **User Query**: Enter a text query or record an audio message.
2. **SQL Generation**: The app generates SQL based on the database schema and user's question.
3. **Query Execution**: Runs the SQL query on a MySQL database and fetches results.
4. **Response**: Provides a natural language response and optional TTS audio.
5. **Logging**: Logs each query and response to the database.
6. **Export**: Download the conversation history as a CSV file.

---
<img width="1440" alt="Screenshot 2024-12-16 at 2 41 01 PM" src="https://github.com/user-attachments/assets/c5a43aec-4d55-4af8-8351-8204017eab2d" />

<img width="1440" alt="Screenshot 2024-12-16 at 3 47 54 PM" src="https://github.com/user-attachments/assets/ec8f9fe5-058d-40ef-985e-de222f2b2b87" />

<img width="1440" alt="Screenshot 2024-12-16 at 2 39 51 PM" src="https://github.com/user-attachments/assets/cead948a-49f0-408c-b4d4-1e205263d6ed" />




# Natural-Language-SQL-Query-Tool

🛠️ Natural Language SQL Query Tool with Streamlit & AWS Bedrock
This application enables users to query MySQL databases using natural language through an intuitive Streamlit interface. Leveraging AWS Bedrock LLMs, this tool transforms user queries into executable SQL and provides responses in natural language, including optional text-to-speech (TTS) output using Amazon Polly.

🚀 Key Features
Natural Language to SQL: Converts user queries into SQL using AWS Bedrock LLMs.
Voice Input Support: Transcribe audio queries using OpenAI's Whisper.
Dynamic MySQL Query Execution: Executes generated SQL queries on a MySQL database.
Natural Language Responses: Returns SQL query results as concise, natural language answers.
TTS Output: Generates audio responses using Amazon Polly.
Chat History Export: Save chat history as a CSV file for easy record-keeping.
Developer Mode: Display generated SQL queries for debugging or learning purposes.
Secure AWS Integration: Configurable AWS credentials for Bedrock and Polly services.
🛠️ Tech Stack
Frontend: Streamlit
Backend: MySQL, AWS Bedrock, Amazon Polly
Libraries:
boto3 for AWS services
whisper for speech-to-text
langchain for LLM-powered SQL generation
streamlit_js_eval for enhanced Streamlit functionality
🔧 How It Works
User Query: Enter a text query or record an audio message.
SQL Generation: The app generates SQL based on the database schema and user's question.
Query Execution: Runs the SQL query on a MySQL database and fetches results.
Response: Provides a natural language response and optional TTS audio.
Logging: Logs each query and response to the database.
Export: Download the conversation history as a CSV file.

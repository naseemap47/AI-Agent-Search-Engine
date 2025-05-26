from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
import streamlit as st


# Wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wapper)

web_search = DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with Search")
"""
In this we are using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit APP.
Examples: [LangChain-Streamlit-Agent](https://python.langchain.com/docs/integrations/callbacks/streamlit/)
"""

## Settings Sidebar
st.sidebar.title("Settings")

# Select Model Type
model_type = st.sidebar.selectbox("Model Type:", ["GROQ", "Ollama"])
if model_type == "GROQ":
    api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")
    selected_model = st.sidebar.selectbox(
        "Select GROQ LLM Model",
        [
            "qwen-qwq-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it",
            "compound-beta", "compound-beta-mini",
            "distil-whisper-large-v3-en", "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile", "llama-guard-3-8b", "llama3-70b-8192", "llama3-8b-8192",
            "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-guard-4-12b", "whisper-large-v3", "whisper-large-v3-turbo",
            "playai-tts", "playai-tts-arabic",
        ],
    )
else:
    selected_model = st.sidebar.selectbox(
        "Select Ollama LLM Model",
        [
            "gemma2:2b", "gemma3:1b", "gemma3", "gemma3:12b", "gemma3:27b",
            "deepseek-r1", "deepseek-r1:671b",
            "llama4:scout", "llama4:maverick",
            "llama3.3", "llama3.2", "llama3.2:1b", "llama3.2-vision", "llama3.2-vision:90b",
            "llama3.1", "llama3.1:405b",
            "qwq", "phi4", "phi4-mini",
            "mistral", "moondream", "neural-chat", "starling-lm", "codellama",
            "llama2-uncensored", "llava", "granite3.3"
        ],
    )


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assisstant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you ?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    if (model_type == "GROQ") and api_key:
        llm = ChatGroq(api_key=api_key, model=selected_model, streaming=True)
        tools = [web_search, arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )
    elif model_type == "Ollama":
        llm = OllamaLLM(model=selected_model)
        tools = [web_search, arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )
    else:
        st.warning("Select LLM Model")
    

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.markdown(response)
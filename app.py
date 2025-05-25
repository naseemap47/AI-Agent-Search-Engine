from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
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
api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

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

    if api_key:
        llm = ChatGroq(api_key=api_key, model="Llama3-8b-8192", streaming=True)
        tools = [web_search, arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handling_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write(response)
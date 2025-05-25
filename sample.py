from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain import hub


# Used in build tool of Wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
# wiki.name

# Used in build tool of Arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
# arxiv.name

web_search = DuckDuckGoSearchRun()

tools = [wiki, arxiv, web_search]

llm = OllamaLLM(model='gemma2:2b')

## Prompt Template
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)


search_agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handling_parsing_errors=True
)

response = search_agent.invoke("What is the capital of France?")
print("Response: ", response)

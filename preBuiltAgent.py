# in this type we are using pre built functions of langchain to create agent

import os
import requests
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent

@tool
def get_weather(city: str):
    """It is used to know weather of the desired city""" #this is doc string it is like a desc for ai to know what this tool does

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    res = requests.get(url)

    data = res.json()

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


@tool
def get_news(city: str):
    """It gives latest news about the desired city"""

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=f"Latest news about {city}",
        search_depth="basic",
        max_results=2
    )

    news_list = []

    for news in response["results"]:
        news_list.append(news["content"])

    return "\n".join(news_list)

llm = ChatMistralAI(model="mistral-small-2506")

# isme all tool calls are decided and executed by agent itself
# in previous approach we did all that manually

agent = create_agent(
    llm,
    tools = [get_weather,get_news],
    system_prompt= "you are a helpful city assistant.",
)

result = agent.invoke({
        "messages": [{"role": "user", "content": "What is weather in mumbai and tell latest news about mumbai"}]
    })


print(result['messages'][-1].content)

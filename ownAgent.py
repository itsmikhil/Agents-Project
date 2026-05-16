import os
import requests
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

# building tools

@tool # this is called decorator this makes function a tool so that it can be invoked
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


# we maintain this list becuase llm tells us which model to use by name
# we need to store it by reference in this list so that it can be easily be called

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}


# initializing llm

llm = ChatMistralAI(model="mistral-small-2506")

# informing llm about our tools 

llm_with_tools = llm.bind_tools([get_weather, get_news])


# it acts as memory for llm to know that next step to take

messages = [
    HumanMessage(
        content="What is weather in Mumbai and tell latest news about Mumbai"
    )
]


# this is first call to llm in this llm descides and tells us which tools should be called

ai_message = llm_with_tools.invoke(messages)

# it should remember what he/she has said otheriwse in every call it will just keep recommending tools
# it is stored inside AIMessage class
messages.append(AIMessage(ai_message))

print("\n===== AI TOOL CALLS =====\n")
print(ai_message.tool_calls)


# based on llm recomendation we call those tools and store their result  under ToolMessage class

for tool_call in ai_message.tool_calls:

    tool_name = tool_call["name"]

    tool_result = tools[tool_name].invoke(tool_call["args"])

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"]
    )

    messages.append(tool_message)


# now we send results of tools back to llm with message history so that it can structure and sent it to user as final result

final_response = llm_with_tools.invoke(messages)

print("\n===== FINAL RESPONSE =====\n")

print(final_response.content)
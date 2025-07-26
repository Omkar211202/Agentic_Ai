import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

model = os.getenv('LLM_MODEL', 'gpt-4o')
weather_api_key = os.getenv("WEATHER_API_KEY")

# Tool 1: Get latitude and longitude from city name
@tool
def get_lat_lon_from_city(city: str):
    """
    Converts a city name to its latitude and longitude using OpenWeatherMap Geocoding API.
    """
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={weather_api_key}"
        response = requests.get(url).json()

        if response.status_code != 200:
            return f"Error getting location: {response.status_code} - {response.text}"

        data = response.json()
        print(data)
        if not data:
            return f"Could not find coordinates for {city}"

        lat = data[0]["lat"]
        lon = data[0]["lon"]
        return json.dumps({"latitude": lat, "longitude": lon})

    except Exception as e:
        return f"Error during geocoding: {str(e)}"

# Tool 2: Get weather from lat/lon
@tool
def get_weather_by_lat_lon(latitude: float, longitude: float):
    """
    Fetches current weather for given coordinates using OpenWeatherMap API.
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={weather_api_key}&units=metric"
        response = requests.get(url)

        if response.status_code != 200:
            return f"Weather API error: {response.status_code} - {response.text}"

        data = response.json()
        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        location = data.get("name", f"({latitude}, {longitude})")

        return f"The current weather in {location} is '{weather}' with a temperature of {temperature}°C."

    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Recursive AI prompting
def prompt_ai(messages, nested_calls=0):
    if nested_calls > 5:
        raise Exception("AI is tool-calling too much!")

    tools = [get_lat_lon_from_city, get_weather_by_lat_lon]
    ai = ChatOpenAI(model=model)
    ai_with_tools = ai.bind_tools(tools)

    ai_response = ai_with_tools.invoke(messages)

    if hasattr(ai_response, "tool_calls") and len(ai_response.tool_calls) > 0:
        messages.append(ai_response)

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Choose tool function
            tool_func = next(t for t in tools if t.name == tool_name)

            print(f"[TOOL INVOKED] {tool_name} with args {tool_args}")
            tool_output = tool_func.invoke(tool_args)
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

        return prompt_ai(messages, nested_calls + 1)

    return ai_response

# Main interaction loop
def main():
    messages = [
        SystemMessage(content=f"You are a weather assistant. You can answer questions about weather for a given city name or latitude/longitude. Today's date is {datetime.now().date()}.")
    ]

    while True:
        user_input = input("Ask about the weather (q to quit): ").strip()
        if user_input.lower() == 'q':
            break

        messages.append(HumanMessage(content=user_input))
        ai_response = prompt_ai(messages)
        print("\nAssistant:", ai_response.content)
        messages.append(ai_response)

if __name__ == "__main__":
    main()

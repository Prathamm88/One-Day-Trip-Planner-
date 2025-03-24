import gradio as gr
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class PlannerState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    city: str
    interests: List[str]
    itinerary: str

lln = ChatGroq(temperature=0,
               groq_api_key="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",
               model_name="llama-3.3-70b-versatile")

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

def travel_planner(city: str, interests: str) -> str:
    # Initialize the state
    state = {
        "messages": [],
        "city": city,
        "interests": [interest.strip() for interest in interests.split(",")],
        "itinerary": ""
    }

    # Adding user message to the state
    state["messages"].append(HumanMessage(content=f"City: {state['city']} Interests: {', '.join(state['interests'])}"))

    # Create itinerary based on user input
    response = lln.invoke(itinerary_prompt.format_messages(city=state["city"], interests=','.join(state['interests'])))

    itinerary = response.content  # Get the content from the response

    # Update state with AI's generated itinerary
    state["messages"].append(AIMessage(content=itinerary))
    state["itinerary"] = itinerary  # Save the itinerary in the state

    return itinerary  # Return the final itinerary

# Gradio Interface
interface = gr.Interface(
    fn=travel_planner,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your Interests (comma-separated)"),
    ],
    outputs=gr.Textbox(label="Generated itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city and your interests to generate a personalized day trip itinerary"
)

interface.launch()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def generate_restaurant_name_and_items(cuisine, model_name, temperature=0.4):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Google API key not found in environment variables")
    
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a creative restaurant naming and menu planning expert."),
        ("human", "Generate a creative restaurant name and 5 menu items for a {cuisine} cuisine restaurant. Return the response with the restaurant name on the first line and menu items as a bulleted list.")
    ])
    
    chain = (
        {"cuisine": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    
    result = chain.invoke(cuisine)
    
    return parse_generation_result(result)

def parse_generation_result(result_text):
    lines = result_text.strip().split('\n')
    restaurant_name = lines[0].replace('#', '').strip()
    
    menu_items = []
    for line in lines[1:]:
        cleaned_line = line.replace('-', '').replace('*', '').strip()
        if cleaned_line:
            menu_items.append(cleaned_line)
    
    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items[:5],
        "menu_items_raw": result_text
    }
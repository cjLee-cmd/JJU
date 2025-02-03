import os
from langchain_teddynote.messages import stream_response
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = '본인의 api_key'

# Define a template with a variable {country}
template = '{country}의 수도는 어디인가요?'

# Create a PromptTemplate object using the from_template method
prompt = PromptTemplate.from_template(template)

# Generate a prompt with the variable {country} replaced
generated_prompt = prompt.format(country="대한민국")
print("Generated Prompt:", generated_prompt)

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    max_tokens=2048,
    temperature=0.1,
)

# Create a chain by combining the prompt and model
chain = prompt | model

# Invoke the chain with a variable
response = chain.invoke("대한민국").content
print("Response:", response)

# Define a template with two variables {country1} and {country2}
template = '{country1}과 {country2}의 수도는 각각 어디인가요?'

# Create a PromptTemplate object with partial variables
prompt = PromptTemplate(
    template=template,
    input_variables=['country1'],
    partial_variables={
        'country2': '미국'  # Set 'country2' as a partial variable
    },
)

# Print the partial template
print("Prompt with Partial Variables:", prompt)

# Format the template with the remaining variable {country1}
formatted_prompt = prompt.format(country1='대한민국')
print("Formatted Prompt:", formatted_prompt)

# Create a partial template where {country2} is provided
prompt_partial = prompt.partial(country2='캐나다')
print("Partial Template with country2='캐나다':", prompt_partial)

# Create a chain using the partial template and the model
chain = prompt_partial | model

# Invoke the chain with both variables
response = chain.invoke({'country1': '대한민국', 'country2': '호주'}).content
print("Response for Updated Variables:", response)
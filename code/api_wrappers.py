import os
import json
import requests

from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from google import genai
from google.genai import types

from openai import AzureOpenAI
from openai import OpenAI

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


#IMPORT SUBSCRIPTION KEYS
import yaml
def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}")
        return None

CODE_DIR = os.path.dirname(__file__)
config = load_config('{}/config.yaml'.format(CODE_DIR))
GOOGLE_API_KEY= config['GOOGLE_API_KEY']
AOAI_SUBSCRIPTION_KEY = config['AOAI_SUBSCRIPTION_KEY']
API_KEY = config['API_KEY']
DASHSCOPE_API_KEY = config['DASHSCOPE_API_KEY']


#-----------------------------------------------
#GEMINI
#-----------------------------------------------
# depeding on model "thinking" mode can't be turned off, in which case the thinking mode will be counted towards your completion
# token budget. 8192 is the default for gemini-2.5-pro where thinking cannot be turned off.
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking#budget
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

def get_geminivisionmodel( system_instruction,
                           contents, 
                           max_completion_tokens=120,
                           thinking_budget=8192,
                           deployment_name='gemini-2.5-pro' ) :
    
    response = genai_client.models.generate_content(
                                    model=deployment_name,
                                    contents=contents,
                                    config=types.GenerateContentConfig(
                                        system_instruction=system_instruction,
                                        max_output_tokens=thinking_budget+max_completion_tokens,
                                        temperature=0.3,
                                    ),
                                )
    return response.text

#-----------------------------------------------
#AOAI - GPT, DEEPSEEKR1 can be called from here
#-----------------------------------------------

from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

import datetime

cred = DefaultAzureCredential()
token = cred.get_token("https://cognitiveservices.azure.com/.default")

endpoint = "YOUR_AOAI_ENDPOINT"
aoai_api_version = "2025-04-01-preview"
aoai_client = AzureOpenAI(
    api_version=aoai_api_version,
    azure_endpoint=endpoint,
    azure_ad_token=token.token,
)

def get_aoai( messages, max_completion_tokens=120, deployment_name='gpt-4.1' ) :
    response = aoai_client.chat.completions.create(
    messages=messages,
            max_completion_tokens=max_completion_tokens,
            model=deployment_name
        )
    return response.choices[0].message.content if len(response.choices)>0 else ''

#-----------------------------------------------
#DEEPSEEK V3
#-----------------------------------------------

def get_deepseek( messages, max_token=120 ) :
    chat_client = ChatCompletionsClient(
        endpoint="YOUR_DEEPSEEK_V3_ENDPOINT",
        credential=AzureKeyCredential(API_KEY),       
    )
    model_name = "DeepSeek-V3-0324-rqlds"
    response = chat_client.complete(
        messages=messages,
        max_tokens=max_token,
        model=model_name
    )
    return response.choices[0].message.content

#-----------------------------------------------
#QWEN-VL-PLUS
#-----------------------------------------------
baba_client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="YOUR_DASHSCOPE_BASE_URL",
    )

def get_baba_response(messages, max_completion_tokens=120,deployment_name="qwen-plus-2025-09-11"):
    response = baba_client.chat.completions.create(
        messages=messages,
        model=deployment_name,
        max_completion_tokens=max_completion_tokens,
        stream=False
        )
    return response.choices[0].message.content if len(response.choices)>0 else ''
from prompts import SYSTEM_PROMPT, INSTRUCTION_PROMPT, TOOLS_PROMPT
import re
from tools import ToolsManager
from llms import LLMManager
import json

def build_prompt(user_input):
    # return f"""
    # {SYSTEM_PROMPT}
    # {INSTRUCTION_PROMPT}
    # {TOOLS_PROMPT}
    # "User query: "{user_input}"
    # """
    system_prompt = SYSTEM_PROMPT + INSTRUCTION_PROMPT + TOOLS_PROMPT
    return system_prompt, user_input

def extract_tools_object(tools_response):
    blocks = re.search(r'<JSON object>(.*)</JSON object>', tools_response, re.DOTALL).group(1)
    return json.loads(blocks)

def run(user_input):
    
    # init all the necessary components
    tools_manager = ToolsManager()
    llm_manager = LLMManager(client_type="azure")
    
    # build the prompt
    system_prompt, user_prompt = build_prompt(user_input)
    
    # call the LLM to determine the tools to use
    tools_response = llm_manager.call_llm(system_prompt, user_prompt, model="gpt-4.1-mini")

    # convert the tools response to a JSON object
    tools_object = extract_tools_object(tools_response)
    
    # print the initial response from the LLM
    print(tools_object['message'])
    
    # approval + apply the tools
    response = ""
    for tool_object in tools_object['actions']:
        # if 'content' in tool_object:
        #     print(tool_object['content'])
        
        tool_name = tool_object['tool']
        parameters = {k: v for k, v in tool_object.items() if k != 'tool'}
        result = tools_manager(tool_name, **parameters)
        response += result

    # return the final result
    return response
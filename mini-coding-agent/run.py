from prompts import SYSTEM_PROMPT, INSTRUCTION_PROMPT, TOOLS_PROMPT, CONTEXT_PROMPT
import re
from tools import ToolsManager
from llms import LLMManager
import json

def build_prompt(context):
    # return f"""
    # {SYSTEM_PROMPT}
    # {INSTRUCTION_PROMPT}
    # {TOOLS_PROMPT}
    # "User query: "{user_input}"
    # """
    system_prompt = SYSTEM_PROMPT + INSTRUCTION_PROMPT + TOOLS_PROMPT + CONTEXT_PROMPT + "\n\nContext: " + json.dumps(context)
    return system_prompt

def run(context):

    # new context of the conversation
    new_context = []
    
    # init all the necessary components
    tools_manager = ToolsManager()
    llm_manager = LLMManager(client_type="azure")
    
    # build the prompt
    system_prompt = build_prompt(context)
    user_prompt = context[-1]["content"] # last user message is the user query
    
    # call the LLM to determine the tools to use 
    # get the response object from the LLM
    response_object, errors = llm_manager.get_response_object(system_prompt, user_prompt, model="gpt-4.1-mini")
    
    # print the initial response from the LLM
    llm_response = response_object['message']
    new_context.append({"role": "assistant", "content": {"response": llm_response, "actions": response_object['actions'], "errors": errors}})
    print(llm_response)
    
    # approval + apply the tools
    for tool_object in response_object['actions']:
        tool_name = tool_object['tool']
        parameters = {k: v for k, v in tool_object.items() if k != 'tool'}
        result = tools_manager(tool_name, **parameters)

        if tool_name != "noop":
            new_context.append({"role": "tool", "content": result})
    
    # return the final result
    return new_context
from openai import OpenAI
from google import genai
from constants import OPENAI_OPENROUTER_BASE_URL, GOOGLE_API_KEY, AZURE_API_KEY, OPENAI_API_KEY, OPENAI_AZURE_BASE_URL
import re
import json
from constants import MAX_RETRIES

class LLMManager:
    def __init__(self, client_type="openai"):
        self.client_type = client_type
        if client_type == "openai":
            self.client = OpenAI(base_url=OPENAI_OPENROUTER_BASE_URL, api_key=OPENAI_API_KEY)
        elif client_type == "google":
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
        elif client_type == "azure":
            self.client = OpenAI(base_url=OPENAI_AZURE_BASE_URL, api_key=AZURE_API_KEY)
        else:
            raise ValueError(f"Invalid client type: {client_type}")

    def call_llm(self, system_prompt, user_prompt, model="liquid/lfm-2.5-1.2b-instruct:free"):
        if self.client_type == "openai":
            return self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}],
                extra_body={"reasoning": {"enabled": False}}
            ).choices[0].message.content
        
        elif self.client_type == "google":
            return self.client.models.generate_content(
                model=model,
                contents=system_prompt + "\n" + user_prompt
            ).text
        
        elif self.client_type == "azure":
            return self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}]
            ).choices[0].message.content
    
    def get_response_object(self, system_prompt, user_prompt, model="liquid/lfm-2.5-1.2b-instruct:free"):
        retries = 0
        errors = {}
        while retries < MAX_RETRIES:
            try:
                response = self.call_llm(system_prompt, user_prompt, model)
                blocks = re.search(r'<answer>(.*)</answer>', response, re.DOTALL).group(1)
                return json.loads(blocks), errors
            except Exception as e:
                retries += 1
                errors[retries] = {"retry_number": retries, "error": str(e), "response": response}
                if retries == MAX_RETRIES:
                    response = {
                        "message": "I'm sorry, I'm having trouble understanding your request. Please try again. Error: " + str(e),
                        "actions": [{"tool": "noop"}]
                    }
                    return response, errors
        return None, {}
import os

OPENAI_API_KEY = '' if os.getenv('OPENAI_API_KEY') is None else os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = '' if os.getenv('GOOGLE_API_KEY') is None else os.getenv('GOOGLE_API_KEY')
AZURE_API_KEY = '' if os.getenv('AZURE_API_KEY') is None else os.getenv('AZURE_API_KEY')
OPENAI_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
OPENAI_AZURE_BASE_URL = '' if os.getenv('OPENAI_AZURE_BASE_URL') is None else os.getenv('OPENAI_AZURE_BASE_URL')
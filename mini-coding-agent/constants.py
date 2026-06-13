from utils.file_utils import set_env_vars

MAX_RETRIES = 3

env_vars = set_env_vars('env.txt')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']
GOOGLE_API_KEY = env_vars['GOOGLE_API_KEY']
AZURE_API_KEY = env_vars['AZURE_API_KEY']
OPENAI_AZURE_BASE_URL = env_vars['OPENAI_AZURE_BASE_URL']
OPENAI_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
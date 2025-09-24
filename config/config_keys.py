import os

try:
    # Load environment variables from a local .env if present (optional)
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Secrets are read from environment variables at import time.
# This keeps API keys out of the repository and supports Docker/Compose and .env files.
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY', '')
COHERE_API_KEY = os.getenv('COHERE_API_KEY', '')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Optional local providers
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

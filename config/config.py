import os

# Provider API selection
# LLM provider: "openrouter" or "ollama"
API = os.getenv("API", "openrouter")

# Embedding provider: "openai" or "ollama"
EMBEDDING_API = os.getenv("EMBEDDING_API", "openai")

# Ollama settings (used when API == "ollama" or EMBEDDING_API == "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EXECUTION_MODEL = os.getenv("OLLAMA_EXECUTION_MODEL", "llama3.1:8b-instruct")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Models (defaults)
# Execution model: used for RAG workflow (generation, question refinement, subquestions, coding)
EXECUTION_MODEL = "qwen/qwen-2.5-7b-instruct"

# Evaluation model: used for judge/evaluator wherever an LLM is needed for scoring
EVAL_MODEL = "qwen/qwen-turbo"

# small model: used for small tasks in the workflow
SMALL_MODEL = "qwen/qwen-2.5-7b-instruct"

# Embedding model name (logical) passed into builder; actual provider chosen by EMBEDDING_API
EMBEDDING_MODEL = "text-embedding-3-small"

BASE_DIR = './' 

INPUT_PATH = os.path.join(BASE_DIR, 'input')  
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')  
# Default storage path derives from current API unless overridden by env below
STORAGE_PATH = os.path.join(BASE_DIR, 'storage', API)

# Maximum number of iterations 
MAX_STEPS = 3

# Disable the second loop (subquestions)
DESABLE_SECOND_LOOP = False

# Max number of PDFs processed in parallel
CONCURRENCY = 3

# Enable evaluation of results
# Determines whether to evaluate the results
EVALUATION = False  

# Use Ragas library for evaluation if EVALUATION = True
# Activates Ragas for evaluation when EVALUATION is enabled
RAGAS = False  

# Use Geval library for evaluation if EVALUATION = True
# Activates Geval for evaluation when EVALUATION is enabled
G_EVAL = False

GROUND_TRUTH = True

# Clear the storage folder before starting
# If set to True, clears the storage folder to ensure a clean run without prior data
CLEAR_STORAGE = False

# Use the Cohere reranker for better context selection
# If True, utilizes the Cohere reranker to improve the selection of the best context
COHERE_RERANK = False

# Subquestion controls
# Maximum number of subquestions to generate per refinement round (1-5)
MAX_SUBQUESTIONS = 3

# CSV export settings (for better compatibility with Excel/Numbers)
# Default to UTF-8 with BOM so Excel detects UTF-8 automatically
CSV_ENCODING = 'utf-8-sig'
# Delimiter: set to ';' for locales where Excel expects semicolon
CSV_DELIMITER = ','

# --- Environment overrides (optional) ---
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y", "on")

INPUT_PATH = os.getenv("INPUT_PATH", INPUT_PATH)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", OUTPUT_PATH)
STORAGE_PATH = os.getenv("STORAGE_PATH", STORAGE_PATH)

_ENV_STORAGE = os.getenv("STORAGE_PATH")
# Provider overrides already applied at top; ensure STORAGE_PATH reflects API
if _ENV_STORAGE is None:
    STORAGE_PATH = os.path.join(BASE_DIR, 'storage', API)
else:
    try:
        last = os.path.basename(os.path.normpath(_ENV_STORAGE))
        if last != API:
            STORAGE_PATH = os.path.join(BASE_DIR, 'storage', API)
    except Exception:
        STORAGE_PATH = os.path.join(BASE_DIR, 'storage', API)

try:
    CONCURRENCY = int(os.getenv("CONCURRENCY", str(CONCURRENCY)))
except Exception:
    pass

EVALUATION = _env_bool("EVALUATION", EVALUATION)
RAGAS = _env_bool("RAGAS", RAGAS)
G_EVAL = _env_bool("G_EVAL", G_EVAL)
CLEAR_STORAGE = _env_bool("CLEAR_STORAGE", CLEAR_STORAGE)
COHERE_RERANK = _env_bool("COHERE_RERANK", COHERE_RERANK)

# CSV overrides via environment
CSV_ENCODING = os.getenv("CSV_ENCODING", CSV_ENCODING)
CSV_DELIMITER = os.getenv("CSV_DELIMITER", CSV_DELIMITER)

# Model overrides via environment (optional)
EXECUTION_MODEL = os.getenv("EXECUTION_MODEL", EXECUTION_MODEL)
EVAL_MODEL = os.getenv("EVAL_MODEL", EVAL_MODEL)
SMALL_MODEL = os.getenv("SMALL_MODEL", SMALL_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
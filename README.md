# OpenAI Proxy Setup Guide

## Dependencies Installation

### Using pip
```bash
pip install fastapi uvicorn openai pydantic pydantic-settings aiosqlite httpx
```

### Using requirements.txt
Create a `requirements.txt` file:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
openai>=1.3.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
aiosqlite>=0.19.0
httpx>=0.25.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Using Poetry (optional)
```bash
poetry add fastapi uvicorn[standard] openai pydantic pydantic-settings aiosqlite httpx
```

## Environment Configuration (.env file)

Create a `.env` file in the same directory as your Python script:

```env
# Required: OpenAI API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Model Override
# OVERRIDE_MODEL=gpt-4

# Optional: Cache Configuration
RESPOND_WITH_CACHE=true
STORE_CACHE=true
CACHE_DIR=./cache

# Optional: Request Timeout (in seconds)
TIMEOUT=30.0
```

## Configuration Options Explained

### Required Settings

- **`OPENAI_BASE_URL`**: The base URL for the OpenAI API
  - Default OpenAI: `https://api.openai.com/v1`
  - For Azure OpenAI: `https://your-resource.openai.azure.com/`
  - For other compatible APIs: Use their base URL

- **`OPENAI_API_KEY`**: Your OpenAI API key
  - Get from: https://platform.openai.com/api-keys
  - Format: `sk-...` (starts with sk-)

### Optional Settings

- **`OVERRIDE_MODEL`**: Force all requests to use a specific model
  - Example: `gpt-4`, `gpt-3.5-turbo`, `claude-3-sonnet`
  - Leave commented out to use the model specified in each request

- **`RESPOND_WITH_CACHE`**: Whether to serve responses from cache
  - `true`: Serve cached responses when available
  - `false`: Always make fresh API calls
  - Default: `false`

- **`STORE_CACHE`**: Whether to store responses in cache
  - `true`: Save API responses to cache database
  - `false`: Don't save responses (cache won't grow)
  - Default: `false`

- **`CACHE_DIR`**: Directory to store the cache database
  - Default: `./cache`
  - The SQLite database will be created at `{CACHE_DIR}/cache.db`

- **`TIMEOUT`**: Request timeout in seconds
  - Default: `30.0`
  - How long to wait for upstream API responses

## Example .env Files

### For Development (with caching)
```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-abc123def456ghi789...
RESPOND_WITH_CACHE=true
STORE_CACHE=true
CACHE_DIR=./dev_cache
TIMEOUT=60.0
```

### For Production/Testing (no caching)
```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-abc123def456ghi789...
RESPOND_WITH_CACHE=false
STORE_CACHE=false
TIMEOUT=30.0
```

### For Azure OpenAI
```env
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment/
OPENAI_API_KEY=your-azure-api-key
RESPOND_WITH_CACHE=true
STORE_CACHE=true
```

### With Model Override
```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-abc123def456ghi789...
OVERRIDE_MODEL=gpt-4
RESPOND_WITH_CACHE=true
STORE_CACHE=true
```

## Running the Proxy

### Start the server
```bash
python openai_proxy.py
```

### Or with uvicorn directly
```bash
uvicorn openai_proxy:app --host 0.0.0.0 --port 8000 --reload
```

### Test the proxy
```bash
# Health check
curl http://localhost:8000/healthz

# Get statistics
curl http://localhost:8000/stats

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## File Structure
```
your-project/
├── openai_proxy.py          # The main proxy script
├── .env                     # Environment configuration
├── requirements.txt         # Python dependencies
└── cache/                   # Cache directory (created automatically)
    └── cache.db            # SQLite cache database
```

## Notes

- The cache directory and database file will be created automatically
- Make sure your `.env` file is not committed to version control
- Add `.env` to your `.gitignore` file
- The proxy runs on `http://localhost:8000` by default
- All cache entries include a session ID for tracking across restarts
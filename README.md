# OpenAI Caching Proxy

This application is a lightweight, self-hosted proxy for the OpenAI API. It provides a simple caching layer that can significantly reduce costs and improve response times for repeated API requests. The caching mechanism focuses only on the chat content to match requests, and it will ignore other metadata or parameters. This allows it to serve cached responses even if minor details in the request, like model id, temperature, etc., are different.

-----

## Setup

### 1\. Install Dependencies

This project uses **uv** for dependency management. If you don't have it installed, you can get it with pip:

```bash
pip install uv
```

Then, install the project dependencies:

```bash
uv sync
```

This will install all packages listed in the `pyproject.toml` file.

### 2\. Configuration

Copy the sample environment file to create your own:

```bash
cp Sample.env .env
```

Open the newly created `.env` file and fill in your OpenAI API key and other settings.

  - **`OPENAI_API_KEY`**: Your OpenAI API key. Needed if you are using OpenAI or any other services that needs api key setup.
  - **`RESPOND_WITH_CACHE`**: Set to `true` to serve cached responses.
  - **`STORE_CACHE`**: Set to `true` to store new responses in the cache.
  - **`OVERRIDE_MODEL`**: Force all requests to use a specific model.

-----

## Running the Proxy

### Start the server

Run the following command from the project root:

```bash
uvicorn openai_proxy:app --host 0.0.0.0 --port 8000 --reload
```

### Test the proxy

You can test the proxy with `curl` commands.

**Health check:**

```bash
curl http://localhost:8000/healthz
```

**Chat completion request:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

-----

## File Structure

```
your-project/
├── openai_proxy.py           # The main proxy script
├── .env.sample               # A sample configuration file
├── .env                      # Your local environment configuration
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Lock file for dependencies
```
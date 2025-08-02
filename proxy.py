#!/usr/bin/env python3
"""
OpenAI-compatible proxy with file-based caching - Improved Version
"""

import json
import hashlib
import uuid
import asyncio
import aiosqlite
import logging
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import openai
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    openai_base_url: str
    openai_api_key: str
    override_model: Optional[str] = None
    respond_with_cache: bool = False
    store_cache: bool = False
    cache_dir: str = "./cache"
    timeout: float = 30.0

    class Config:
        env_file = ".env"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, list]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = None


settings = Settings()

# Generate unique session ID for this application run
SESSION_ID = str(uuid.uuid4())[:8]
SESSION_START_TIME = datetime.utcnow().isoformat() + "Z"

# Configure OpenAI client (sync version for model listing)
openai_client = openai.OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.timeout,
)

# Create async HTTP client for cancellable requests
async_http_client = None

# Global lock for cache operations to prevent race conditions
cache_lock = asyncio.Lock()

# Track active requests for proper cleanup
active_requests = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_http_client

    # Create cache directory and initialize database
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
    await init_cache_db()

    # Initialize async HTTP client for cancellable requests
    async_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.timeout),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
    )

    logger.info("OpenAI Proxy started with:")
    logger.info(f"  Session ID: {SESSION_ID}")
    logger.info(f"  Base URL: {settings.openai_base_url}")
    logger.info(f"  Override Model: {settings.override_model}")
    logger.info(f"  Cache DB: {get_cache_db_path()}")
    logger.info(f"  Respond with Cache: {settings.respond_with_cache}")
    logger.info(f"  Store Cache: {settings.store_cache}")

    yield

    # Cleanup active requests on shutdown
    for request_id, task in active_requests.items():
        logger.info(f"Cancelling active request {request_id} on shutdown")
        task.cancel()

    # Close async HTTP client
    if async_http_client:
        await async_http_client.aclose()


app = FastAPI(lifespan=lifespan)


def trim_messages(messages: list) -> list:
    """Trim whitespace from message content for canonicalization"""
    trimmed = []
    for msg in messages:
        if isinstance(msg, dict):
            trimmed_msg = msg.copy()
            if "content" in trimmed_msg and isinstance(trimmed_msg["content"], str):
                trimmed_msg["content"] = trimmed_msg["content"].strip()
            trimmed.append(trimmed_msg)
        else:
            trimmed.append(msg)
    return trimmed


def canonicalize_request(request_data: dict, endpoint_type: str) -> dict:
    """Create canonical request for cache key generation"""
    canonical = {}

    # Apply model override
    model = settings.override_model or request_data.get("model")
    canonical["model"] = model

    # Add endpoint-specific fields
    if endpoint_type == "chat":
        canonical["messages"] = trim_messages(request_data.get("messages", []))
    elif endpoint_type == "completions":
        canonical["prompt"] = request_data.get("prompt")

    # Add generation parameters if present
    params = [
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "n",
    ]
    for param in params:
        if param in request_data:
            canonical[param] = request_data[param]

    return canonical


def generate_cache_key(canonical_request: dict) -> str:
    """Generate SHA256 cache key from canonical request"""
    json_str = json.dumps(canonical_request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_cache_db_path() -> Path:
    """Get cache database file path"""
    return Path(settings.cache_dir) / "cache.db"


async def init_cache_db():
    """Initialize SQLite cache database"""
    db_path = get_cache_db_path()

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                canonical_request TEXT NOT NULL,
                original_request TEXT NOT NULL,
                response TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                session_id TEXT NOT NULL
            )
        """)

        # Add indexes for better performance
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_created ON cache_entries(session_id, created_at)"
        )

        await db.commit()


async def load_cached_response(cache_key: str) -> Optional[dict]:
    """Load cached response from database"""
    if not settings.respond_with_cache:
        return None

    db_path = get_cache_db_path()

    try:
        async with cache_lock:  # Prevent race conditions
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row:
                        logger.info(f"Cache HIT for key: {cache_key[:12]}...")
                        return {
                            "canonical_request": json.loads(row["canonical_request"]),
                            "original_request": json.loads(row["original_request"]),
                            "response": json.loads(row["response"]),
                            "metadata": {
                                "created_at": row["created_at"],
                                "updated_at": row["updated_at"],
                                "cache_key": row["cache_key"],
                                "status_code": row["status_code"],
                                "session_id": row["session_id"],
                            },
                        }

    except Exception as e:
        logger.error(f"Error loading from cache: {e}")

    return None


async def store_cached_response(
    cache_key: str,
    canonical_request: dict,
    original_request: dict,
    response: dict,
    status_code: int,
):
    """Store response in cache database (updates existing entries)"""
    if not settings.store_cache:
        return

    db_path = get_cache_db_path()
    now = datetime.utcnow().isoformat() + "Z"

    try:
        async with cache_lock:  # Prevent race conditions
            async with aiosqlite.connect(db_path) as db:
                # Check if entry exists
                async with db.execute(
                    "SELECT cache_key FROM cache_entries WHERE cache_key = ?",
                    (cache_key,),
                ) as cursor:
                    exists = await cursor.fetchone() is not None

                if exists:
                    # Update existing entry
                    await db.execute(
                        """
                        UPDATE cache_entries 
                        SET canonical_request = ?, original_request = ?, response = ?, 
                            status_code = ?, updated_at = ?, session_id = ?
                        WHERE cache_key = ?
                    """,
                        (
                            json.dumps(canonical_request),
                            json.dumps(original_request),
                            json.dumps(response),
                            status_code,
                            now,
                            SESSION_ID,
                            cache_key,
                        ),
                    )
                    logger.info(f"Updated cache entry for key: {cache_key[:12]}...")
                else:
                    # Insert new entry
                    await db.execute(
                        """
                        INSERT INTO cache_entries 
                        (cache_key, canonical_request, original_request, response, 
                         status_code, created_at, updated_at, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            cache_key,
                            json.dumps(canonical_request),
                            json.dumps(original_request),
                            json.dumps(response),
                            status_code,
                            now,
                            now,
                            SESSION_ID,
                        ),
                    )
                    logger.info(f"Stored new cache entry for key: {cache_key[:12]}...")

                await db.commit()

    except Exception as e:
        logger.error(f"Error storing cache: {e}")


async def monitor_client_disconnect(request: Request, request_id: str) -> bool:
    """Monitor for client disconnect"""
    try:
        while True:
            if await request.is_disconnected():
                logger.info(f"Client disconnected for request {request_id}")
                return True
            await asyncio.sleep(0.1)  # Check every 100ms
    except asyncio.CancelledError:
        logger.debug(f"Disconnect monitoring cancelled for request {request_id}")
        return False
    except Exception as e:
        logger.error(f"Error monitoring disconnect for request {request_id}: {e}")
        return False


async def forward_request(
    request_data: dict,
    endpoint_type: str,
    request_id: str,
    disconnect_event: asyncio.Event,
) -> tuple[dict, int]:
    """Forward request to upstream OpenAI API with proper cancellation support"""
    # Apply model override
    if settings.override_model:
        request_data = request_data.copy()
        request_data["model"] = settings.override_model

    try:
        logger.info(
            f"[{request_id}] Forwarding {endpoint_type} request to: {settings.openai_base_url}"
        )
        logger.info(f"[{request_id}] Request model: {request_data.get('model')}")

        # Check if client disconnected before making upstream call
        if disconnect_event.is_set():
            logger.info(
                f"[{request_id}] Client disconnected, aborting upstream request"
            )
            raise HTTPException(status_code=499, detail="Client disconnected")

        # Determine the endpoint URL
        if endpoint_type == "chat":
            url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        elif endpoint_type == "completions":
            url = f"{settings.openai_base_url.rstrip('/')}/completions"
        else:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")

        # Create the upstream request task using httpx for proper cancellation
        upstream_task = asyncio.create_task(
            async_http_client.post(url, json=request_data)
        )

        # Create a task that waits for disconnect
        disconnect_task = asyncio.create_task(disconnect_event.wait())

        try:
            # Wait for either the upstream response or client disconnect
            done, pending = await asyncio.wait(
                [upstream_task, disconnect_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check what completed first
            if disconnect_event.is_set():
                logger.info(
                    f"[{request_id}] Client disconnected during upstream request - request CANCELLED"
                )
                # The upstream_task cancellation will actually cancel the HTTP request
                raise HTTPException(status_code=499, detail="Client disconnected")

            # Get the upstream response
            http_response = await upstream_task

            # Check HTTP status
            if http_response.status_code != 200:
                logger.error(
                    f"[{request_id}] Upstream API returned status {http_response.status_code}"
                )
                try:
                    error_data = http_response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Unknown error"
                    )
                except Exception as e:
                    error_msg = f"HTTP {http_response.status_code}\n{str(e)}"
                raise HTTPException(
                    status_code=502, detail=f"Upstream API error: {error_msg}"
                )

            response_data = http_response.json()
            logger.info(
                f"[{request_id}] Successfully completed {endpoint_type} request"
            )
            return response_data, 200

        except asyncio.CancelledError:
            logger.info(
                f"[{request_id}] Request was cancelled (likely due to client disconnect)"
            )
            raise HTTPException(status_code=499, detail="Request cancelled")

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except httpx.TimeoutException as e:
        logger.error(f"[{request_id}] Timeout error: {e}")
        raise HTTPException(status_code=504, detail="Upstream request timed out")
    except httpx.RequestError as e:
        logger.error(f"[{request_id}] Request error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream request error: {e}")
    except asyncio.CancelledError:
        logger.info(f"[{request_id}] Request cancelled")
        raise HTTPException(status_code=499, detail="Request cancelled")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        logger.error(f"[{request_id}] Error type: {type(e)}")
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


async def handle_request(request: Request, endpoint_type: str):
    """Common request handling logic"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting {endpoint_type} request")

    # Set up client disconnect detection
    disconnect_event = asyncio.Event()
    disconnect_task = asyncio.create_task(
        monitor_client_disconnect(request, request_id)
    )

    # Track this request
    active_requests[request_id] = disconnect_task

    try:
        request_data = await request.json()

        # Set disconnect event when client disconnects
        async def set_disconnect_on_completion():
            await disconnect_task
            disconnect_event.set()

        disconnect_completion_task = asyncio.create_task(set_disconnect_on_completion())

        # Canonicalize request and generate cache key
        canonical_request = canonicalize_request(request_data, endpoint_type)
        cache_key = generate_cache_key(canonical_request)
        effective_model = canonical_request["model"]

        logger.info(f"[{request_id}] Cache key: {cache_key[:12]}...")

        # Try cache first
        cached_entry = await load_cached_response(cache_key)
        if cached_entry:
            logger.info(
                f"[{request_id}] Serving from cache (created: {cached_entry['metadata']['created_at']})"
            )
            response = JSONResponse(cached_entry["response"])
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Model-Used"] = effective_model
            response.headers["X-Request-ID"] = request_id
            return response

        logger.info(f"[{request_id}] Cache MISS - forwarding to upstream API")

        # Forward to upstream
        response_data, status_code = await forward_request(
            request_data, endpoint_type, request_id, disconnect_event
        )

        # Store in cache if successful
        if status_code == 200:
            logger.info(f"[{request_id}] Storing response in cache")
            await store_cached_response(
                cache_key, canonical_request, request_data, response_data, status_code
            )

        response = JSONResponse(response_data, status_code=status_code)
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Model-Used"] = effective_model
        response.headers["X-Request-ID"] = request_id
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in request handling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up
        if request_id in active_requests:
            del active_requests[request_id]

        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass

        try:
            disconnect_completion_task.cancel()
            await disconnect_completion_task
        except (asyncio.CancelledError, NameError):
            pass

        logger.info(f"[{request_id}] Request completed")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await handle_request(request, "chat")


@app.post("/v1/completions")
async def completions(request: Request):
    return await handle_request(request, "completions")


@app.get("/v1/models")
async def list_models():
    try:
        response = openai_client.models.list()
        return response.model_dump()
    except openai.APITimeoutError:
        raise HTTPException(status_code=504, detail="Upstream request timed out")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


@app.get("/healthz")
async def health_check():
    active_count = len(active_requests)
    return {
        "status": "ok",
        "session_id": SESSION_ID,
        "active_requests": active_count,
        "cache_enabled": settings.respond_with_cache,
    }


@app.get("/stats")
async def get_stats():
    """Get cache and request statistics"""
    try:
        db_path = get_cache_db_path()
        async with aiosqlite.connect(db_path) as db:
            # Get total cache entries
            async with db.execute(
                "SELECT COUNT(*) as total FROM cache_entries"
            ) as cursor:
                total_row = await cursor.fetchone()
                total_entries = total_row[0] if total_row else 0

            # Get entries for current session
            async with db.execute(
                "SELECT COUNT(*) as session_total FROM cache_entries WHERE session_id = ?",
                (SESSION_ID,),
            ) as cursor:
                session_row = await cursor.fetchone()
                session_entries = session_row[0] if session_row else 0

        return {
            "session_id": SESSION_ID,
            "session_start": SESSION_START_TIME,
            "active_requests": len(active_requests),
            "cache_stats": {
                "total_entries": total_entries,
                "session_entries": session_entries,
                "respond_with_cache": settings.respond_with_cache,
                "store_cache": settings.store_cache,
            },
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": "Could not retrieve stats"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
src/dataset_generation/azure_client.py
──────────────────────────────────────
Thin, stateless wrapper around the Azure OpenAI SDK.
Credentials are loaded from environment variables; never from arguments.
"""

import time
import logging
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from src.config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

logger = logging.getLogger(__name__)

_client: AzureOpenAI | None = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
    return _client


def validate_azure_connection() -> tuple[bool, str]:
    """
    Validate that the configured Azure deployment is reachable.
    Returns (ok, message). Never exposes the API key.
    """
    from src.config import azure_configured, azure_config_errors
    if not azure_configured():
        missing = azure_config_errors()
        return False, f"Missing environment variables: {', '.join(missing)}"
    try:
        client = get_client()
        # Minimal probe — list models if supported, otherwise send a tiny chat
        client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True, f"Connected to deployment '{AZURE_DEPLOYMENT}' successfully."
    except RateLimitError:
        return False, "Azure rate limit reached. Please wait and retry."
    except APIConnectionError as exc:
        return False, f"Connection error: {_safe_exc(exc)}"
    except APITimeoutError:
        return False, "Request timed out. Check your AZURE_OPENAI_ENDPOINT."
    except Exception as exc:
        return False, f"Azure validation failed: {_safe_exc(exc)}"


def chat_completion(
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    json_mode: bool = False,
    retries: int = 4,
) -> str:
    """
    Call the configured Azure deployment and return the response text.
    Retries with exponential backoff on rate-limit and transient errors.
    Raises RuntimeError with a safe (no-credential) message on failure.
    """
    client = get_client()
    kwargs: dict = dict(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except RateLimitError as exc:
            wait = 2 ** attempt * 5
            logger.warning("Rate limit hit (attempt %d/%d), waiting %ds", attempt + 1, retries, wait)
            time.sleep(wait)
            last_error = exc
        except (APIConnectionError, APITimeoutError) as exc:
            wait = 2 ** attempt * 2
            logger.warning("Transient error (attempt %d/%d): %s", attempt + 1, retries, _safe_exc(exc))
            time.sleep(wait)
            last_error = exc
        except Exception as exc:
            raise RuntimeError(f"Azure API error: {_safe_exc(exc)}") from exc

    raise RuntimeError(
        f"Azure API failed after {retries} attempts: {_safe_exc(last_error)}"
    )


def _safe_exc(exc: Exception) -> str:
    """Return exception string with API key scrubbed."""
    msg = str(exc)
    if AZURE_API_KEY:
        msg = msg.replace(AZURE_API_KEY, "***")
    return msg

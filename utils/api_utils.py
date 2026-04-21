"""
API call utility functions
"""
import asyncio
import aiohttp
import base64
import os
from typing import Optional
from config import API_KEY, API_URL, API_MODEL, API_TIMEOUT, TEMPERATURE, MAX_TOKENS, REQUEST_DELAY

# Image encoding cache to avoid re-encoding the same image
_image_cache = {}

# 429 retry configuration
MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # Initial retry wait in seconds, exponential backoff


def _encode_image(image_path: str) -> str:
    """
    Encode an image to base64 (with cache).

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image data
    """
    if not image_path or not os.path.exists(image_path):
        return ""

    if image_path in _image_cache:
        return _image_cache[image_path]

    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            _image_cache[image_path] = encoded
            return encoded
    except Exception as e:
        print(f"Image encoding failed {image_path}: {e}")
        return ""

async def fetch_api(
    session: aiohttp.ClientSession,
    prompt: str,
    image_path: Optional[str] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    timeout: int = API_TIMEOUT,
    request_delay: float = REQUEST_DELAY
) -> str:
    """
    Async API call (supports text and image, OpenAI-compatible format, with 429 retry)
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Build message content
    if image_path and os.path.exists(image_path):
        image_data = _encode_image(image_path)
        if not image_data:
            content = prompt
        else:
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
    else:
        content = prompt

    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    api_endpoint = API_URL

    for attempt in range(MAX_RETRIES + 1):
        try:
            await asyncio.sleep(request_delay)

            async with session.post(api_endpoint, headers=headers, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        return f"Unexpected API response format: {str(result)[:200]}"
                elif response.status == 429:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    return f"API rate limited (429), failed after {MAX_RETRIES} retries"
                elif response.status == 401:
                    return f"API authentication failed (401), please check API Key"
                else:
                    error_text = await response.text()
                    return f"API request failed (status {response.status}): {error_text[:200]}"
        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(wait)
                continue
            return f"API call timed out (>{timeout}s), failed after {MAX_RETRIES} retries"
        except Exception as e:
            return f"API call exception: {str(e)[:200]}"

    return "API call failed: exceeded max retries"

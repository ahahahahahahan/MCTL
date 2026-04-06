"""
API 调用工具函数
"""
import asyncio
import aiohttp
import base64
import os
from typing import Optional
from config import API_KEY, API_URL, API_MODEL, API_TIMEOUT, TEMPERATURE, MAX_TOKENS, REQUEST_DELAY

# 图片编码缓存，避免重复编码同一张图片
_image_cache = {}

# 429 重试配置
MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # 首次重试等待秒数，指数退避


def _encode_image(image_path: str) -> str:
    """
    将图片编码为base64（带缓存）

    Args:
        image_path: 图片文件路径

    Returns:
        base64 编码的图片数据
    """
    if not image_path or not os.path.exists(image_path):
        return ""

    # 使用缓存避免重复编码
    if image_path in _image_cache:
        return _image_cache[image_path]

    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            _image_cache[image_path] = encoded
            return encoded
    except Exception as e:
        print(f"图片编码失败 {image_path}: {e}")
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
    异步调用 API（支持文本和图片，OpenAI 兼容格式，含 429 重试）
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # 构建消息内容
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
                        return f"API响应格式异常: {str(result)[:200]}"
                elif response.status == 429:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    return f"API请求频率过高(429)，已重试{MAX_RETRIES}次仍失败"
                elif response.status == 401:
                    return f"API认证失败(401)，请检查API Key"
                else:
                    error_text = await response.text()
                    return f"API请求失败(状态码{response.status}): {error_text[:200]}"
        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(wait)
                continue
            return f"API调用超时(>{timeout}秒)，已重试{MAX_RETRIES}次"
        except Exception as e:
            return f"API调用异常: {str(e)[:200]}"

    return "API调用失败: 超出最大重试次数"

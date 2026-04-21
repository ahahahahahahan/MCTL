"""
Text processing utility functions
"""


def extract_answer(response: str) -> str:
    """
    Extract answer from response.
    Returns: "real", "fake", or "unknown"
    """
    response_lower = response.lower()

    # English format
    if "answer:" in response_lower:
        answer = response_lower.split("answer:")[-1].split('.')[0].strip().strip('[').strip(']').strip('"')
        return answer

    # Chinese format (simplified + traditional) as fallback
    for marker in ["答案：", "答案:"]:
        if marker in response:
            answer = response.split(marker)[-1].split('.')[0].split('。')[0].strip().strip('[').strip(']').strip('"')
            return answer

    # Direct keyword search
    if "fake" in response_lower:
        return "fake"
    if "real" in response_lower:
        return "real"
    if "虚假" in response or "虛假" in response:
        return "fake"
    if "真实" in response or "真實" in response:
        return "real"

    return "unknown"


def normalize_prediction(answer: str) -> int:
    """
    Normalize answer to numeric value.
    Returns: 1 (fake) or 0 (real), None if unrecognizable
    """
    if not answer:
        return None
    answer_lower = answer.lower().strip()
    if answer_lower in ("fake", "false", "harmful", "虚假", "虛假"):
        return 1
    if answer_lower in ("real", "true", "not harmful", "harmless", "真实", "真實"):
        return 0
    # Fuzzy match
    if "fake" in answer_lower or "虚假" in answer_lower or "虛假" in answer_lower:
        return 1
    if "real" in answer_lower or "真实" in answer_lower or "真實" in answer_lower:
        return 0
    return None

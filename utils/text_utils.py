"""
文本处理工具函数
"""


def extract_answer(response: str) -> str:
    """
    从响应中提取答案
    返回: "真实", "虚假", "real", "fake" 或 "未知"
    """
    response_lower = response.lower()

    # 中文格式
    if "答案：" in response:
        answer = response.split("答案：")[-1].split('.')[0].strip().strip('[').strip(']').strip('"')
        return answer
    # 英文格式
    if "answer:" in response_lower:
        answer = response_lower.split("answer:")[-1].split('.')[0].strip().strip('[').strip(']').strip('"')
        return answer

    # 直接查找关键词
    if "虚假" in response:
        return "虚假"
    if "真实" in response:
        return "真实"
    if "fake" in response_lower:
        return "fake"
    if "real" in response_lower:
        return "real"

    return "未知"


def normalize_prediction(answer: str) -> int:
    """
    将答案标准化为数值
    返回: 1 (虚假/fake) 或 0 (真实/real)，无法识别返回 None
    """
    if not answer:
        return None
    answer_lower = answer.lower().strip()
    if answer_lower in ("虚假", "fake", "false", "harmful"):
        return 1
    if answer_lower in ("真实", "real", "true", "not harmful", "harmless"):
        return 0
    # 模糊匹配
    if "虚假" in answer_lower or "fake" in answer_lower:
        return 1
    if "真实" in answer_lower or "real" in answer_lower:
        return 0
    return None

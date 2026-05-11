"""
logging_utils.py — 타이밍 포맷.
"""


def format_elapsed(seconds: float) -> str:
    """초(float) → '시간 분 초' 문자열."""
    total = max(0.0, float(seconds))
    minutes, rem = divmod(total, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours}시간 {minutes:02d}분 {rem:05.2f}초"
    if minutes > 0:
        return f"{minutes}분 {rem:05.2f}초"
    return f"{rem:.2f}초"

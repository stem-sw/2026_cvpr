"""
json_utils.py — 모델 출력에서 JSON 객체를 안정적으로 추출.
"""
import json
import re
from typing import Any, Dict, List, Optional

# extract_first_json_object에서 유효 dict로 인정할 키 목록
_VALID_KEYS = {"candidates", "accident_time", "frame_index", "center_x", "center_y",
               "type", "is_single", "confidence", "evidence"}


def strip_thinking_text(text: str) -> str:
    if not text:
        return text
    text = str(text).replace('\ufeff', '')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'^(?:The user wants|I understand|Okay|Let me|Based on).*?:\s*',
                  '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_json_candidate(candidate: str) -> List[str]:
    base = str(candidate or '').strip()
    if not base:
        return []

    variants: List[str] = []

    def add(v: str) -> None:
        v = str(v or '').strip()
        if v and v not in variants:
            variants.append(v)

    def merge_adjacent(v: str) -> str:
        pat = re.compile(r'"((?:[^"\\]|\\.)*)"\s+"((?:[^"\\]|\\.)*)"')
        prev = None
        while v != prev:
            prev = v
            v = pat.sub(lambda m: '"' + m.group(1).rstrip() + ' ' + m.group(2).lstrip() + '"', v)
        return v

    add(base)
    add(base.replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2018', "'").replace('\u2019', "'"))

    for seed in list(variants):
        fixed = seed
        fixed = re.sub(r'^```(?:json)?\s*', '', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\s*```$', '', fixed)
        fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'\bNaN\b', 'null', fixed)
        fixed = re.sub(r'\bInfinity\b', '999999', fixed)
        fixed = re.sub(r',\s*""\s*([}\]])', r'\1', fixed)
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        fixed = merge_adjacent(fixed)
        add(fixed)

        if '\\"' in fixed:
            unescaped = fixed.replace('\\"', '"')
            add(merge_adjacent(unescaped))

    return variants


def try_parse_single_json(candidate: str) -> Optional[Dict[str, Any]]:
    for variant in normalize_json_candidate(candidate):
        try:
            obj = json.loads(variant)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    # 닫는 괄호 누락 보완
    for variant in normalize_json_candidate(candidate):
        if variant.strip().startswith('{'):
            for i in range(1, 4):
                try:
                    obj = json.loads(variant.strip() + "}" * i)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
    return None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = strip_thinking_text(text or '')
    if not text:
        return None

    text = re.sub(r'```(?:json)?', '', text, flags=re.IGNORECASE)
    text = text.replace('```', '')
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bNaN\b', 'null', text)
    text = re.sub(r'\bInfinity\b', '999999', text)

    stack, candidates = [], []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            candidates.append(text[stack.pop(): i + 1])

    last_open = text.rfind('{')
    if last_open != -1:
        candidates.append(text[last_open:])

    for cand in reversed(candidates):
        parsed = try_parse_single_json(cand)
        if parsed and _VALID_KEYS & set(parsed.keys()):
            return parsed

    return None

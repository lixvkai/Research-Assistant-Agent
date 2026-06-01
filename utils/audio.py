"""语音转文字 — 使用 OpenAI-compatible Whisper 或本地降级方案。"""

import os


def transcribe_audio(audio_path: str) -> str:
    """将音频文件转为文字。优先使用 DeepSeek/OpenAI Whisper API，失败则降级。"""
    if not audio_path or not os.path.exists(audio_path):
        return ""

    # 尝试使用 OpenAI Whisper API（DeepSeek 不支持则可换成 OpenAI）
    try:
        from openai import OpenAI
        from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
        return result.text
    except Exception:
        pass

    # 降级：使用本地 transformers pipeline（如果安装了 whisper）
    try:
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
        result = pipe(audio_path)
        return result["text"]
    except Exception:
        pass

    return "[语音识别失败：请确保安装了 transformers 和 torch，或配置有效的 Whisper API]"

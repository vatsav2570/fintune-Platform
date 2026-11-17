"""
src/inference/chat_engine.py
─────────────────────────────
Generate responses from the loaded model.
Supports streaming via transformers TextIteratorStreamer.
"""

from __future__ import annotations
import logging
import threading
from typing import Iterator, Optional

from src.inference.model_loader import get_loaded

logger = logging.getLogger(__name__)


def generate_response(
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    Non-streaming generation. Returns the full response string.
    messages: list of {"role": ..., "content": ...}
    """
    try:
        import torch
    except ImportError:
        return "⚠️ PyTorch is not installed in the active environment. Please install it with: pip install torch"

    model, tokenizer = get_loaded()
    if model is None:
        return "⚠️ No model loaded. Please load a model from the Fine-Tuned Models page."

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def generate_streaming(
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> Iterator[str]:
    """
    Streaming generation. Yields token-by-token text deltas.
    Compatible with Streamlit's st.write_stream.
    """
    try:
        import torch
        from transformers import TextIteratorStreamer
    except ImportError as exc:
        yield f"⚠️ Required package missing: {exc}. Please install PyTorch and transformers (pip install torch transformers)."
        return

    model, tokenizer = get_loaded()
    if model is None:
        yield "⚠️ No model loaded. Please load a model from the Fine-Tuned Models page."
        return

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        do_sample=do_sample,
        top_p=top_p if do_sample else 1.0,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    def _generate():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_generate)
    thread.start()

    for token in streamer:
        yield token

    thread.join()

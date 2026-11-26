"""SSE collection for --stream (cameralogbench.eval.runner._read_sse_chat)."""
import json

from cameralogbench.eval.runner import _read_sse_chat


def _lines(*chunks):
    out = []
    for c in chunks:
        out.append(("data: " + (c if isinstance(c, str) else json.dumps(c)) + "\n").encode())
        out.append(b"\n")
    return out


def test_stream_collects_content_finish_and_usage():
    d = _read_sse_chat(_lines(
        {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
        {"choices": [{"delta": {"content": '{"choice": '}}]},
        {"choices": [{"delta": {"content": '"C"}'}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        "[DONE]"))
    assert d["choices"][0]["message"]["content"] == '{"choice": "C"}'
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}


def test_stream_reasoning_only_gives_empty_answer():
    d = _read_sse_chat(_lines({"choices": [{"delta": {"reasoning_content": "x"}, "finish_reason": "length"}]}, "[DONE]"))
    assert d["choices"][0]["message"]["content"] is None
    assert d["choices"][0]["finish_reason"] == "length"


def test_stream_error_chunk_and_empty_stream():
    assert "error" in _read_sse_chat(_lines({"error": {"message": "boom"}}))
    assert _read_sse_chat([b": keep-alive\n", b"\n"])["choices"] == []

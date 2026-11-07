"""Shared verification budget and provider usage accounting, including retries."""
import json
import os
import threading
from types import SimpleNamespace


class BudgetExceeded(RuntimeError):
    pass


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class UsageLedger:
    def __init__(self, max_total_tokens: int | None = None, max_calls: int | None = None):
        if max_total_tokens is None:
            max_total_tokens = int(os.getenv("LLM_BUDGET_MAX_TOKENS", "1000000"))
        if max_calls is None:
            max_calls = int(os.getenv("LLM_BUDGET_MAX_CALLS", "100"))
        if max_total_tokens <= 0 or max_calls <= 0:
            raise ValueError("验证总 token 上限和调用次数必须为正整数")
        self.max_total_tokens, self.max_calls = max_total_tokens, max_calls
        self.records = []
        self.charged_tokens = 0
        self.exhausted = False
        self.lock = threading.Lock()

    def reserve(self, kwargs):
        # UTF-8 byte count plus framing is a deliberately conservative input
        # reservation, not an assertion about actual tokenizer/provider usage.
        prompt = json.dumps({k: v for k, v in kwargs.items() if k not in {
            "max_tokens", "max_completion_tokens", "timeout", "stream", "stream_options"}}, ensure_ascii=False)
        framing = 256 * (len(kwargs.get("messages") or []) + 1)
        output_limit = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if not isinstance(output_limit, int) or output_limit <= 0:
            raise BudgetExceeded("受控验证要求每次调用设置输出 token 上限")
        reserved = len(prompt.encode("utf-8")) + framing + output_limit
        with self.lock:
            if self.exhausted or len(self.records) >= self.max_calls or self.charged_tokens + reserved > self.max_total_tokens:
                self.exhausted = True
                raise BudgetExceeded("验证总额度不足，已停止发送新请求")
            record = {"call": len(self.records) + 1, "model": kwargs.get("model"), "reserved_tokens": reserved,
                      "charged_tokens": reserved, "status": "in_progress", "usage": None,
                      "usage_complete": False}
            self.records.append(record)
            self.charged_tokens += reserved
        return record

    def finish(self, record, usage, error=None):
        prompt = _value(usage, "prompt_tokens")
        completion = _value(usage, "completion_tokens")
        total = _value(usage, "total_tokens")
        complete = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in (prompt, completion, total))
        complete = complete and total == prompt + completion
        with self.lock:
            if record["status"] != "in_progress":
                return
            record["status"] = "failed" if error else "completed"
            record["error_type"] = type(error).__name__ if error else None
            if complete:
                raw = usage.model_dump() if hasattr(usage, "model_dump") else usage if isinstance(usage, dict) else {
                    "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
                record.update(usage=raw, usage_complete=True, charged_tokens=total)
                self.charged_tokens += total - record["reserved_tokens"]
                if self.charged_tokens > self.max_total_tokens:
                    self.exhausted = True
            else:
                record["usage_note"] = "供应方未返回完整用量；实际值未知，保守占用完整预留额度。"

    def snapshot(self):
        with self.lock:
            records = [dict(row) for row in self.records]
            return {"max_total_tokens": self.max_total_tokens, "max_calls": self.max_calls,
                    "charged_tokens": self.charged_tokens, "exhausted": self.exhausted,
                    "actual_usage_complete": all(r["usage_complete"] for r in records),
                    "reported_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records if r["usage_complete"]),
                    "reported_completion_tokens": sum(r["usage"]["completion_tokens"] for r in records if r["usage_complete"]),
                    "reported_total_tokens": sum(r["usage"]["total_tokens"] for r in records if r["usage_complete"]),
                    "calls": records}


class TrackingClient:
    """Wrap all SDK completion calls of a verification client in one shared ledger."""
    def __init__(self, client, ledger):
        # SDK transport retries would bypass this ledger; workflow retries remain
        # visible as separate create calls and are charged separately.
        configure = getattr(client, "with_options", None)
        self.client = configure(max_retries=0) if callable(configure) else client
        self.ledger = ledger
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __getattr__(self, key):
        return getattr(self.client, key)

    def create(self, **kwargs):
        if kwargs.get("stream"):
            kwargs["stream_options"] = {**(kwargs.get("stream_options") or {}), "include_usage": True}
        record = self.ledger.reserve(kwargs)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.ledger.finish(record, None, exc)
            raise
        if not kwargs.get("stream"):
            self.ledger.finish(record, _value(response, "usage"))
            return response

        def tracked_stream():
            usage, error = None, None
            try:
                for chunk in response:
                    usage = _value(chunk, "usage") or usage
                    yield chunk
            except BaseException as exc:
                error = exc
                raise
            finally:
                self.ledger.finish(record, usage, error)
                close = getattr(response, "close", None)
                if close:
                    close()
        return tracked_stream()

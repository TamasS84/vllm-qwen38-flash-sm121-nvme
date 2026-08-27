#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
import urllib.request

DEFAULT_MAX_MODEL_LEN = 262144
DEFAULT_COMPLETION_TOKENS = 128


def request_json(
    url: str,
    payload: dict[str, object] | None = None,
    *,
    timeout: float,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_context_contract(
    *,
    configured_max_model_len: int,
    expected_max_model_len: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if configured_max_model_len != expected_max_model_len:
        raise ValueError(
            "server reported max_model_len "
            f"{configured_max_model_len}, expected {expected_max_model_len}"
        )
    requested_tokens = prompt_tokens + completion_tokens
    if requested_tokens != expected_max_model_len:
        raise ValueError(
            f"requested token total {requested_tokens} does not exactly fill "
            f"the expected {expected_max_model_len}-token context"
        )


def validate_usage(
    usage: object,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if not isinstance(usage, dict):
        raise RuntimeError("response did not include token usage")
    if usage.get("prompt_tokens") != prompt_tokens:
        raise RuntimeError(
            "response prompt token count does not match the requested count: "
            f"{usage.get('prompt_tokens')} != {prompt_tokens}"
        )
    if usage.get("completion_tokens") != completion_tokens:
        raise RuntimeError(
            "response completion token count does not match the requested count: "
            f"{usage.get('completion_tokens')} != {completion_tokens}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an exact near-limit Qwen3.8 context request."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN - DEFAULT_COMPLETION_TOKENS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_COMPLETION_TOKENS,
    )
    parser.add_argument(
        "--expected-max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/v1/models", timeout=args.timeout)
    model = models["data"][0]["id"]

    tokenized = request_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "prompt": "Stable context marker. ",
            "add_special_tokens": False,
        },
        timeout=args.timeout,
    )
    marker_tokens = tokenized["tokens"]
    max_model_len = tokenized["max_model_len"]
    prompt_tokens = args.prompt_tokens
    validate_context_contract(
        configured_max_model_len=max_model_len,
        expected_max_model_len=args.expected_max_model_len,
        prompt_tokens=prompt_tokens,
        completion_tokens=args.max_tokens,
    )
    if not marker_tokens:
        raise ValueError("context marker produced no tokens")

    repeats = (prompt_tokens + len(marker_tokens) - 1) // len(marker_tokens)
    prompt = (marker_tokens * repeats)[:prompt_tokens]
    started = time.perf_counter()
    response = request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started
    usage = response.get("usage")
    validate_usage(
        usage,
        prompt_tokens=prompt_tokens,
        completion_tokens=args.max_tokens,
    )
    print(
        json.dumps(
            {
                "configured_max_model_len": max_model_len,
                "requested_prompt_tokens": prompt_tokens,
                "requested_completion_tokens": args.max_tokens,
                "elapsed_seconds": elapsed,
                "usage": usage,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

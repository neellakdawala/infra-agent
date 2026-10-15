"""Render the exact text Hermes 3 sees for a tool-calling request (no model weights downloaded)."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")

tools = [{
    "type": "function",
    "function": {
        "name": "get_pod_logs",
        "description": "Fetch recent logs for a Kubernetes pod. Use when asked why a pod is failing or restarting.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "pod": {"type": "string", "description": "Pod name"},
            },
            "required": ["namespace", "pod"],
        },
    },
}]

messages = [
    {"role": "system", "content": "You are an infrastructure triage assistant."},
    {"role": "user", "content": "The pod payments-7f9c in namespace prod keeps restarting. Why?"},
]

prompt = tok.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,   # append "<|im_start|>assistant\n" so the model starts its reply
    tokenize=False,               # return the string, not token IDs, so we can read it
)
print(prompt)
print(f"\n--- {len(tok(prompt).input_ids)} tokens ---")
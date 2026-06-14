import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class Model:
    def __init__(self, checkpoint: str, device: str = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        dtype = torch.float32 if self.device == "cpu" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype).to(self.device)
        self.model.eval()

    def _render(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def get_logits(self, prompt: str):
        text = self._render(prompt)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**model_inputs)
        return output.logits[0, -1, :]

    def get_hidden_states(self, prompt: str) -> torch.Tensor:
        """Residual-stream activation at the last position,
        for every layer."""
        text = self._render(prompt)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**model_inputs, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1) of [batch, seq, hidden]
        last = torch.stack([h[0, -1, :] for h in output.hidden_states])
        return last.float().cpu()

    def score_continuation(self, prompt: str, continuation: str) -> float:
        """Mean per-token log P(continuation | prompt) under teacher forcing."""
        prompt_ids = self.tokenizer(self._render(prompt), add_special_tokens=False).input_ids
        cont_ids = self.tokenizer(continuation, add_special_tokens=False).input_ids
        input_ids = torch.tensor([prompt_ids + cont_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[0].float()
        len_p, len_c = len(prompt_ids), len(cont_ids)
        pred = logits[len_p - 1 : len_p + len_c - 1]
        logprobs = torch.log_softmax(pred, dim=-1)
        target = torch.tensor(cont_ids, device=self.device)
        token_lp = logprobs[torch.arange(len_c), target]
        return token_lp.mean().item()

    def best_continuation(self, prompt: str, variants) -> float:
        """Most probable continuation among variants."""
        return max(self.score_continuation(prompt, v) for v in variants)

    def generate_start(self, prompt, n=15):
        text = self._render(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=n, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

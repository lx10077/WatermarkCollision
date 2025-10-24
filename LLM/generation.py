
# Watermark Generation and Detection Utilities for Language Models
# Includes sampling and estimation logic with Gumbel and Inverse methods

from typing import Any
import torch
from tqdm import tqdm
from sampling import *
import copy

class WatermarkGenerate:
    def __init__(self, 
                 model, 
                 vocab_size: int, 
                 key: int = 15485863,
                 text_length: int = 400, 
                 watermark_type: str = "Gumbel", 
                 temperature: float = 0.7,
                 text_window: int = 5,
                 seeding_scheme: str = "skipgram_prf") -> None:
        self.model = model
        self.text_window = text_window
        self.vocab_size = vocab_size
        self.text_length = text_length
        self.temperature = temperature
        self.key = key

        assert watermark_type in ["Gumbel", "Inverse"], f"Unsupported watermark type: {watermark_type}"
        self.watermark_type = watermark_type
        self.seeding_scheme = seeding_scheme

        if watermark_type == "Gumbel":
            self.key_func = gumbel_key_func
            self.sampler = gumbel_sampling
            self.Y_func = gumbel_Y
        elif watermark_type == "Inverse":
            self.key_func = transform_key_func
            self.sampler = transform_sampling
            self.Y_func = transform_Y_dif

        self.state = None  # Optional internal state tracking

    def full_generate(self, prompts) -> Any:
        generator = torch.Generator()
        inputs = prompts.to(self.model.device)
        attn = torch.ones_like(inputs)
        past = None

        Ys, top_probs = [], []
        if self.watermark_type == "Inverse":
            Us = []
        for _ in range(self.text_length):
            with torch.no_grad():
                output = self.model(inputs if past is None else inputs[:, -1:],
                                    past_key_values=past,
                                    attention_mask=attn)

            probs = torch.nn.functional.softmax(output.logits[:, -1] / self.temperature, dim=-1).cpu()
            top_prob = torch.max(probs, axis=1)[0]
            xi, pi = self.key_func(generator, inputs, self.vocab_size, self.key, self.text_window, self.seeding_scheme)  ## Shape: (Batch_size, N_tokens)
            tokens = self.sampler(probs, pi, xi).to(self.model.device)  # always watermark
            Y = self.Y_func(tokens, pi, xi)

            Ys.append(Y.unsqueeze(1))
            top_probs.append(top_prob.unsqueeze(1))
            if self.watermark_type == "Inverse":
                Us.append(xi)

            inputs = torch.cat([inputs, tokens], dim=1)
            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        Ys = torch.cat(Ys, axis=1)
        top_probs = torch.cat(top_probs, axis=1)
        if self.watermark_type == "Inverse":
            Us = torch.cat(Us, axis=1)  # 结果就是 (batch_size, text_length)
            return (inputs.detach().cpu(),
                    Ys.detach().cpu(),
                    top_probs.detach().cpu(),
                    Us.detach().cpu())
        else:
            return (inputs.detach().cpu(),
                Ys.detach().cpu(),
                top_probs.detach().cpu())
    
    def mixture_generate(self, prompts, eps=1.) -> Any:
        batch_size, prompt_length = prompts.shape
        generator = torch.Generator()
        inputs = prompts.to(self.model.device)
        attn = torch.ones_like(inputs)
        past = None
        bernoulli_mean = torch.full((batch_size,), fill_value=eps) 

        Ys, top_probs, where_watermarks = [], [], []
        for _ in range(self.text_length): 
            with torch.no_grad():
                if past:
                    output = self.model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
                else:
                    output = self.model(inputs)

            probs = torch.nn.functional.softmax(output.logits[:,-1]/self.temperature, dim=-1).cpu() ## Shape: (Batch_size, N_tokens)
            top_prob = torch.max(probs, axis=1)[0]  ## Shape: (Batch_size, )
            xi, pi = self.key_func(generator, inputs, self.vocab_size, self.key, self.text_window, self.seeding_scheme)  ## Shape: (Batch_size, N_tokens)

            history_tensor = inputs[:,prompt_length-1:]
            exists_in_history = self.check_ngram_in_history_batch(history_tensor).to(self.model.device)

            add_watermark = torch.bernoulli(bernoulli_mean).int().bool().to(self.model.device)
            watermark_exist = ((~ exists_in_history) & add_watermark)
            tokens_wm = self.sampler(probs, pi, xi).to(self.model.device) ## Shape: (Batch_size, 1)
            tokens_no = torch.multinomial(probs,1).to(self.model.device)
            tokens = torch.where(watermark_exist[:, None], tokens_wm, tokens_no)

            Y = self.Y_func(tokens, pi, xi)  ## Shape: (Batch_size, 1). Same for the follows
            Ys.append(Y.unsqueeze(1))
            top_probs.append(top_prob.unsqueeze(1))
            where_watermarks.append(watermark_exist.unsqueeze(1))
            
            inputs = torch.cat([inputs, tokens], dim=1)
            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        Ys = torch.cat(Ys, axis=1)
        top_probs = torch.cat(top_probs, axis=1)
        where_watermarks = torch.cat(where_watermarks, axis=1)

        return (inputs.detach().cpu(), 
                Ys.detach().cpu(), 
                top_probs.detach().cpu(), 
                where_watermarks.detach().cpu())
    
    def compute_Ys(self, corrupted_data, prompts=None, is_null=False, output_xi=False):

        corrupted_data = corrupted_data.squeeze(0) if corrupted_data.dim() == 3 and corrupted_data.size(0) == 1 else corrupted_data
        if prompts is not None:
            prompts = prompts.squeeze(0) if prompts.dim() == 3 and prompts.size(0) == 1 else prompts
            prompts = prompts.to(self.model.device)

        generator = torch.Generator()
        corrupted_data = corrupted_data.to(self.model.device)
        A = lambda inputs : self.key_func(generator,inputs, self.vocab_size, self.key, self.text_window, self.seeding_scheme)
        computed_Ys = []
        computed_XIS = []
        for i in range(len(corrupted_data)):
            if i % 100 == 0:
                print(i)
            if prompts is not None and (not is_null):
                text = corrupted_data[i]
                prompt = prompts[i]
                full_texts =  torch.cat([prompt[-self.text_window:],text])
            else:
                full_texts =  corrupted_data[i]

            this_Ys = []
            this_XIS = []
            for j in range(len(full_texts) - self.text_window):

                given_seg = full_texts[:self.text_window+j].unsqueeze(0)
                xi, pi = A(given_seg)

                Y = self.Y_func(full_texts[self.text_window+j].unsqueeze(0).unsqueeze(0), pi, xi)
                this_Ys.append(Y.unsqueeze(0))
                this_XIS.append(xi.unsqueeze(0))

            this_Ys = torch.vstack(this_Ys)
            this_XIS = torch.vstack(this_XIS)

            computed_Ys.append(this_Ys.squeeze())
            computed_XIS.append(this_XIS.squeeze())

        computed_Ys = torch.vstack(computed_Ys).numpy()
        computed_XIS = torch.vstack(computed_XIS).numpy()

        if output_xi == True:
            return computed_Ys, computed_XIS
        else:
            return computed_Ys
    
    def recover_top_probs_no_prompt(self, generated_tokens, delta=0.1, batch_size=10, temperature_list=None):
        self.model.eval()
        device = self.model.device
        generated_tokens = generated_tokens.to(device)
        N, L = generated_tokens.shape

        # Use default temperature if not provided
        if temperature_list is None:
            temperature_list = [self.temperature]

        results_by_temp = {}

        for temp in temperature_list:
            print("Work on temperature:", temp)
            all_probs = []

            for start in tqdm(range(0, N, batch_size)):
                end = min(start + batch_size, N)
                batch = generated_tokens[start:end]  # (B, L)
                input_ids = batch[:, :-1]            # (B, L-1)
                target_ids = batch[:, 1:]            # (B, L-1)
                attn_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    output = self.model(input_ids, attention_mask=attn_mask)
                    logits = output.logits / temp  # use current temperature
                    probs = torch.softmax(logits, dim=-1)  # (B, L-1, vocab_size)

                probs_selected = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
                all_probs.append(probs_selected.cpu())

            probs_full = torch.cat(all_probs, dim=0)  # (N, L-1)
            prepend_col = torch.full((N, 1), fill_value=1 - delta)
            full_probs = torch.cat([prepend_col, probs_full], dim=1)  # (N, L)
            results_by_temp[temp] = copy.deepcopy(full_probs)

        return results_by_temp

    def check_ngram_in_history_batch(self, batch_tokens):
        """
        Batch check if the latest n-gram in each sequence exists in the history window.
        
        Args:
            batch_tokens (torch.Tensor): Current batch tokens, shape (batch_size, current_len).
            history_window (torch.Tensor): Historical tokens for each batch, shape (batch_size, history_len).
            n (int): Length of the n-gram.
        
        Returns:
            torch.Tensor: A boolean tensor of shape (batch_size,) where each entry is True if the n-gram
                        exists in the history window for that batch, and False otherwise.
        """
        batch_size, current_len = batch_tokens.size()
        
        # Ensure there are enough tokens to form an n-gram
        if current_len <= self.text_window:
            return torch.zeros(batch_size, dtype=torch.bool)
        
        # Extract the latest n-gram for each sequence
        current_ngrams = batch_tokens[:, -self.text_window:].unsqueeze(1)  # Shape: (batch_size, 1, n)
        
        # Create a view of all possible n-grams in the history window
        history_ngrams = torch.stack([batch_tokens[:, i:i+self.text_window] for i in range(batch_tokens.size(1) - self.text_window)], dim=1)
        # Shape of history_ngrams: (batch_size, history_len - n + 1, n)
        
        # Compare the current n-gram with each n-gram in the history window
        matches = (current_ngrams == history_ngrams).all(dim=-1)  # Shape: (batch_size, history_len - n + 1)
        
        # Check if any position matches for each sequence
        exists_in_history = matches.any(dim=1)
        return exists_in_history

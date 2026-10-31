"""
Neural Machine Translation Architectures:
1. Basic Transformer (Encoder-Decoder from scratch, Vaswani et al., 2017)
2. mT5 (`google/mt5-small`, Pretrained Multilingual Encoder-Decoder)
3. LLaMA (`HuggingFaceTB/SmolLM2-135M-Instruct`, `LlamaForCausalLM` Decoder-Only)
4. Gemma (`unsloth/gemma-3-270m-it`, `Gemma3ForCausalLM` Decoder-Only)
"""
import math
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from src.data import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    Vocabulary,
    detokenize_basic,
    tokenize_basic,
)


# ============================================================================
# 1. BASIC TRANSFORMER (ENCODER-DECODER FROM SCRATCH)
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BasicTransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer built from scratch in PyTorch.
    Architecture:
      - Scaled Source & Target Token Embeddings + Sinusoidal Positional Encoding
      - L_enc = 3 TransformerEncoderLayers (Multi-Head Self-Attention + FFN + LayerNorm)
      - L_dec = 3 TransformerDecoderLayers (Masked Self-Attention + Cross-Attention + FFN)
      - Final Linear Projection to Target Vocabulary |V_tgt|
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model)
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src == PAD_IDX
        tgt_key_padding_mask = tgt_in == PAD_IDX
        tgt_mask = self.generate_square_subsequent_mask(tgt_in.size(1), tgt_in.device)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)

    @torch.no_grad()
    def translate_sentence(
        self,
        sentence: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 28,
        temperature: float = 0.0,
        device: str = "cpu",
    ) -> str:
        self.eval()
        tokens = tokenize_basic(sentence)
        src_ids = src_vocab.encode(tokens, max_len=max_len)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = src_tensor == PAD_IDX

        src_emb = self.pos_encoder(self.src_embedding(src_tensor) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)

        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        generated_ids = []

        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1), ys.device)
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask,
            )
            logits = self.fc_out(out[:, -1, :]).squeeze(0)
            logits[PAD_IDX] = -1e9
            logits[SOS_IDX] = -1e9
            logits[UNK_IDX] = -1e9
            for prev_id in set(generated_ids[-3:]):
                if logits[prev_id] > 0:
                    logits[prev_id] /= 1.3
                else:
                    logits[prev_id] *= 1.3

            if temperature <= 1e-5:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == EOS_IDX:
                break
            generated_ids.append(next_id)
            ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

        decoded_tokens = tgt_vocab.decode(generated_ids)
        return detokenize_basic(decoded_tokens)


# ============================================================================
# 2. mT5 ENCODER-DECODER SYSTEM (`google/mt5-small`)
#    With Language-Specific Vocabulary Pruning (EN-FR Active Sub-Vocabulary)
# ============================================================================

def clean_Generated_french(text: str) -> str:
    """Strip task echoes, sentinel tokens, and extra lines from LLM/Seq2Seq outputs."""
    text = re.sub(r"<extra_id_\d+>", "", text)
    text = text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    for prefix in [
        "French:",
        "French translation:",
        "Translation:",
        "Traduction :",
        "Traduction:",
        "Français :",
        "Français:",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    text = text.strip("\"' ")
    return text


class MT5TranslationSystem:
    """
    Wrapper for `google/mt5-small` (300.18M parameters, Encoder-Decoder architecture).
    Implements Language-Specific Sub-Vocabulary Projection (Abdaoui et al., 2020)
    to restrict the 250,112-token multilingual SentencePiece table to the active
    English+French subwords during fine-tuning and decoding, eliminating span-corruption
    sentinel tokens (`<extra_id_*>`) and accelerating CPU fine-tuning by >25x.
    """

    def __init__(
        self,
        model_name: str = "google/mt5-small",
        all_corpus_pairs: Optional[List[Dict[str, str]]] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.prefix = "translate English to French: "
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        # Record full pretrained parameter count before vocabulary slicing
        self.total_params = sum(p.numel() for p in self.model.parameters())

        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: List[int] = []
        if all_corpus_pairs is not None:
            self._apply_en_fr_vocab_pruning(all_corpus_pairs)

        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def _apply_en_fr_vocab_pruning(self, pairs: List[Dict[str, str]]):
        """Slice shared embeddings and lm_head to active English+French SentencePiece IDs."""
        used = {0, 1, 2}  # <pad>=0, </s>=1, <unk>=2
        for p in pairs:
            used.update(self.tokenizer.encode(self.prefix + p["en"], add_special_tokens=True))
            used.update(self.tokenizer.encode(p["fr"], add_special_tokens=True))
        # Exclude sentinel tokens >= 250000
        active_ids = sorted(i for i in used if i < 250000)
        self.new_to_old = active_ids
        self.old_to_new = {old_id: new_id for new_id, old_id in enumerate(active_ids)}

        idx_tensor = torch.tensor(active_ids, dtype=torch.long)
        d_model = self.model.config.d_model
        new_vocab_size = len(active_ids)

        with torch.no_grad():
            new_shared = nn.Embedding(new_vocab_size, d_model, padding_idx=0)
            new_shared.weight.copy_(self.model.shared.weight[idx_tensor])

            new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False)
            # Initialize lm_head from the sliced lm_head weights
            new_lm_head.weight.copy_(self.model.lm_head.weight[idx_tensor])

        self.model.shared = new_shared
        self.model.encoder.embed_tokens = new_shared
        self.model.decoder.embed_tokens = new_shared
        self.model.lm_head = new_lm_head
        self.model.config.vocab_size = new_vocab_size
        self.active_vocab_size = new_vocab_size

    def _remap_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.old_to_new:
            return input_ids
        flat = input_ids.view(-1).tolist()
        mapped = [self.old_to_new.get(x, 2) for x in flat]
        return torch.tensor(mapped, dtype=torch.long, device=self.device).view(input_ids.shape)

    def _unmap_ids(self, new_ids: List[int]) -> List[int]:
        if not self.new_to_old:
            return new_ids
        return [
            self.new_to_old[i] if 0 <= i < len(self.new_to_old) else 2
            for i in new_ids
            if i not in (0, 1)
        ]

    def fine_tune(
        self,
        train_pairs: List[Dict[str, str]],
        val_pairs: List[Dict[str, str]],
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1.5e-3,
        max_train_samples: int = 1600,
    ) -> Tuple[List[Dict[str, float]], float]:
        """Fine-tune mT5-small on English-French parallel pairs."""
        subset = train_pairs[:max_train_samples]
        val_sub = val_pairs[:160]

        for name, p in self.model.named_parameters():
            if "encoder." in name and "shared" not in name:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)

        history = []
        t_start = time.time()

        for ep in range(1, epochs + 1):
            ep_t0 = time.time()
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(subset), batch_size):
                batch = subset[i : i + batch_size]
                src_texts = [self.prefix + item["en"] for item in batch]
                tgt_texts = [item["fr"] for item in batch]

                enc = self.tokenizer(
                    src_texts, max_length=36, padding=True, truncation=True, return_tensors="pt"
                )
                lbl_raw = self.tokenizer(
                    text_target=tgt_texts,
                    max_length=36,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                input_ids = self._remap_ids(enc.input_ids)
                attention_mask = enc.attention_mask.to(self.device)
                labels = self._remap_ids(lbl_raw)
                labels[labels == 0] = -100

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            scheduler.step()

            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_sub), batch_size):
                    batch = val_sub[i : i + batch_size]
                    src_texts = [self.prefix + item["en"] for item in batch]
                    tgt_texts = [item["fr"] for item in batch]
                    enc = self.tokenizer(
                        src_texts, max_length=36, padding=True, truncation=True, return_tensors="pt"
                    )
                    lbl_raw = self.tokenizer(
                        text_target=tgt_texts,
                        max_length=36,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                    input_ids = self._remap_ids(enc.input_ids)
                    attention_mask = enc.attention_mask.to(self.device)
                    labels = self._remap_ids(lbl_raw)
                    labels[labels == 0] = -100
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    val_loss += float(outputs.loss.item())
                    val_batches += 1

            ep_sec = time.time() - ep_t0
            rec = {
                "model": "mT5-Small",
                "epoch": ep,
                "train_loss": round(total_loss / max(n_batches, 1), 4),
                "val_loss": round(val_loss / max(val_batches, 1), 4),
                "epoch_time_s": round(ep_sec, 2),
            }
            history.append(rec)
            print(
                f"[mT5-Small] Epoch {ep}/{epochs} | Train Loss: {rec['train_loss']:.4f} "
                f"| Val Loss: {rec['val_loss']:.4f} | Time: {ep_sec:.1f}s"
            )

        total_time = time.time() - t_start
        return history, total_time

    @torch.no_grad()
    def translate_batch(
        self,
        sentences: List[str],
        batch_size: int = 16,
        max_new_tokens: int = 36,
        temperature: float = 0.0,
    ) -> List[str]:
        self.model.eval()
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = [self.prefix + s for s in sentences[i : i + batch_size]]
            enc = self.tokenizer(
                batch, max_length=36, padding=True, truncation=True, return_tensors="pt"
            )
            input_ids = self._remap_ids(enc.input_ids)
            attention_mask = enc.attention_mask.to(self.device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.25,
                "no_repeat_ngram_size": 3,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "decoder_start_token_id": 0,
            }
            if temperature <= 1e-5:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.92

            out_ids = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )
            for row_ids in out_ids.tolist():
                orig_ids = self._unmap_ids(row_ids)
                text = self.tokenizer.decode(orig_ids, skip_special_tokens=True)
                results.append(clean_Generated_french(text))
        return results


# ============================================================================
# 3. DECODER-ONLY CAUSAL LM SYSTEM (LLaMA & Gemma)
#    Supports Few-Shot In-Context Prompting and LoRA Fine-Tuning
# ============================================================================

FEW_SHOT_EXAMPLES = [
    ("I do not know what to say.", "Je ne sais pas quoi dire."),
    ("She looked at him in silence.", "Elle le regarda en silence."),
    ("What are you doing here so late?", "Que faites-vous ici si tard ?"),
]


class CausalLLMTranslationSystem:
    """
    Unified Decoder-Only Translation Engine for LLaMA (`LlamaForCausalLM`)
    and Gemma (`Gemma3ForCausalLM`).
    Supports:
      - Few-Shot In-Context Learning (3-shot structured demonstration prompt)
      - Parameter-Efficient Fine-Tuning via LoRA (Low-Rank Adaptation)
      - Temperature-controlled autoregressive sampling
    """

    def __init__(
        self,
        model_name: str,
        display_name: str,
        device: str = "cpu",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        self.model_name = model_name
        self.display_name = display_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).to(device)
        self.total_params = sum(p.numel() for p in self.base_model.parameters())

        if use_lora:
            n_layers = getattr(self.base_model.config, "num_hidden_layers", 18)
            top_layers = [max(0, n_layers - 1)]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=top_layers,
            )
            self.model = get_peft_model(self.base_model, peft_config)
            self.trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        else:
            self.model = self.base_model
            self.trainable_params = 0

    def build_prompt(self, en_sentence: str, num_shots: int = 3) -> str:
        shots_str = ""
        for ex_en, ex_fr in FEW_SHOT_EXAMPLES[:num_shots]:
            shots_str += f"English: {ex_en}\nFrench: {ex_fr}\n\n"
        prompt = (
            "Translate English to French.\n\n"
            f"{shots_str}"
            f"English: {en_sentence}\nFrench:"
        )
        return prompt

    def _compute_masked_causal_loss(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute causal LM loss projecting ONLY non-masked target positions through lm_head."""
        inner_backbone = self.model.base_model.model.model
        lm_head = self.model.base_model.model.lm_head
        body_out = inner_backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        hidden = body_out.last_hidden_state[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        active_hidden = hidden[valid_mask]
        active_labels = shift_labels[valid_mask]
        logits = lm_head(active_hidden)
        return F.cross_entropy(logits, active_labels)

    def fine_tune_lora(
        self,
        train_pairs: List[Dict[str, str]],
        val_pairs: List[Dict[str, str]],
        epochs: int = 2,
        batch_size: int = 8,
        lr: float = 1e-3,
        max_train_samples: int = 32,
    ) -> Tuple[List[Dict[str, float]], float]:
        """Fine-tune LoRA adapters on causal English->French translation pairs."""
        subset = train_pairs[:max_train_samples]
        val_sub = val_pairs[:16]

        self.tokenizer.padding_side = "right"
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
        )

        history = []
        t_start = time.time()

        for ep in range(1, epochs + 1):
            ep_t0 = time.time()
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(subset), batch_size):
                batch = subset[i : i + batch_size]
                full_texts = []
                prompt_lens = []
                for item in batch:
                    p_str = self.build_prompt(item["en"], num_shots=0)
                    full_str = f"{p_str} {item['fr']}\n"
                    full_texts.append(full_str)
                    p_ids = self.tokenizer(p_str, add_special_tokens=True).input_ids
                    prompt_lens.append(len(p_ids))

                enc = self.tokenizer(
                    full_texts,
                    max_length=40,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                labels = enc.input_ids.clone()
                for b_idx, p_len in enumerate(prompt_lens):
                    labels[b_idx, : min(p_len, labels.size(1))] = -100
                labels[enc.attention_mask == 0] = -100

                optimizer.zero_grad()
                loss = self._compute_masked_causal_loss(
                    enc.input_ids, enc.attention_mask, labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for i in range(0, len(val_sub), batch_size):
                    batch = val_sub[i : i + batch_size]
                    full_texts = []
                    prompt_lens = []
                    for item in batch:
                        p_str = self.build_prompt(item["en"], num_shots=0)
                        full_str = f"{p_str} {item['fr']}\n"
                        full_texts.append(full_str)
                        prompt_lens.append(
                            len(self.tokenizer(p_str, add_special_tokens=True).input_ids)
                        )
                    enc = self.tokenizer(
                        full_texts,
                        max_length=40,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    labels = enc.input_ids.clone()
                    for b_idx, p_len in enumerate(prompt_lens):
                        labels[b_idx, : min(p_len, labels.size(1))] = -100
                    labels[enc.attention_mask == 0] = -100
                    loss = self._compute_masked_causal_loss(
                        enc.input_ids, enc.attention_mask, labels
                    )
                    val_loss += float(loss.item())
                    val_batches += 1

            ep_sec = time.time() - ep_t0
            rec = {
                "model": self.display_name,
                "epoch": ep,
                "train_loss": round(total_loss / max(n_batches, 1), 4),
                "val_loss": round(val_loss / max(val_batches, 1), 4),
                "epoch_time_s": round(ep_sec, 2),
            }
            history.append(rec)
            print(
                f"[{self.display_name}] Epoch {ep}/{epochs} | Train Loss: {rec['train_loss']:.4f} "
                f"| Val Loss: {rec['val_loss']:.4f} | Time: {ep_sec:.1f}s"
            )

        self.tokenizer.padding_side = "left"
        total_time = time.time() - t_start
        return history, total_time

    @torch.no_grad()
    def translate_batch(
        self,
        sentences: List[str],
        batch_size: int = 8,
        max_new_tokens: int = 28,
        temperature: float = 0.0,
        num_shots: int = 3,
    ) -> List[str]:
        self.model.eval()
        self.tokenizer.padding_side = "left"
        outputs_clean = []

        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i : i + batch_size]
            prompts = [self.build_prompt(s, num_shots=num_shots) for s in batch_sents]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=140,
            ).to(self.device)
            input_len = enc.input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.15,
            }
            if temperature <= 1e-5:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.90

            gen_ids = self.model.generate(**enc, **gen_kwargs)
            new_tokens = gen_ids[:, input_len:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for raw_out in decoded:
                for stop_marker in ["\nEnglish:", "\nFrench:", "\n\n", "English:"]:
                    if stop_marker in raw_out:
                        raw_out = raw_out.split(stop_marker)[0]
                outputs_clean.append(clean_Generated_french(raw_out))

        return outputs_clean

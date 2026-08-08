"""Batched CTC inference for the w2v-BERT arms, with optional blank-penalty decoding.

Logits are computed once per arm and decoded once per penalty, because the expensive part is the
forward pass. Sweeping five penalties therefore costs barely more than decoding once, which is why
the ensemble can afford so many members.
"""

from __future__ import annotations

from pathlib import Path

from waxal_asr.audio import load_clip
from waxal_asr.decode import greedy_decode


def transcribe_ctc(
    model_id: str,
    clip_ids: list[str],
    audio_dir: Path,
    penalty: float = 0.0,
    batch_size: int = 8,
    sample_rate: int = 16000,
    use_attention_mask: bool = True,
) -> dict[str, str]:
    """Transcribe every clip with one CTC checkpoint.

    Args:
        model_id: Hugging Face repository id, or a local directory.
        clip_ids: identifiers to transcribe.
        audio_dir: directory holding one audio file per identifier.
        penalty: amount subtracted from the CTC blank logit before the arg max.
        batch_size: clips per forward pass.
        sample_rate: input sample rate.
        use_attention_mask: whether to tell the encoder which frames are padding. Correct is True.
            False exists only to reproduce the blank-penalty members of the submitted ensemble,
            which were decoded by a script that omitted the mask; see the note below.

    Returns:
        Mapping of clip identifier to transcript.
    """
    import torch
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

    processor = Wav2Vec2BertProcessor.from_pretrained(model_id)
    model = Wav2Vec2BertForCTC.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    blank_id = model.config.pad_token_id

    predictions: dict[str, str] = {}
    for start in range(0, len(clip_ids), batch_size):
        batch = clip_ids[start : start + batch_size]
        waveforms = [load_clip(cid, audio_dir, sample_rate) for cid in batch]
        inputs = processor(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        # Clips in a batch have different lengths, so the shorter ones are zero-padded. With the
        # mask, self-attention ignores that padding. Without it, the padding is treated as real
        # audio and the encoder output changes for every frame, not just the padded tail: it alters
        # roughly half of all transcripts. Masking is therefore the correct behaviour and the
        # default here.
        #
        # The submitted ensemble's blank-penalty members were nonetheless produced by a script that
        # omitted the mask, so reproducing those members requires omitting it too. That is recorded
        # per member in configs/ensembles.yaml rather than hidden in this file.
        if use_attention_mask:
            model_inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            model_inputs = {"input_features": inputs["input_features"].to(device)}
        with torch.no_grad():
            logits = model(**model_inputs).logits.float().cpu()
        for offset, clip_id in enumerate(batch):
            predictions[clip_id] = greedy_decode(logits[offset], processor, blank_id, penalty)
        if start % (batch_size * 20) == 0:
            print(f"[ctc] {min(start + batch_size, len(clip_ids))}/{len(clip_ids)}", flush=True)
    return predictions

"""Transcription for the fine-tuned Whisper arm.

The Whisper checkpoint is a sequence-to-sequence model, so it cannot be decoded the way the CTC arms
are: there is no per-frame arg max, and no blank symbol to penalise. It generates.

Decoding follows the arm's own config of record (`configs/turbo_linsna_r.yaml`): greedy, that is
``num_beams: 1``, and ``max_new_tokens: 200``. No language token is forced, because that config sets
``infer.language_from_id: false`` and the Phase 2 identifiers carry no language.

Clips are passed to the feature extractor as they are, which means Whisper's fixed 30 second
receptive field truncates the 3.3 per cent of clips that run longer. That is what produced the
submitted member, so it is what this function does. The third-party Sunbird arm in
``waxal_asr.modeling.sunbird`` does window long audio, and the two arms differ in this deliberately.
"""

from __future__ import annotations

from pathlib import Path

from waxal_asr.audio import load_clip

NUM_BEAMS = 1
MAX_NEW_TOKENS = 200


def transcribe_whisper(
    model_id: str,
    clip_ids: list[str],
    audio_dir: Path,
    batch_size: int = 8,
    sample_rate: int = 16000,
) -> dict[str, str]:
    """Transcribe every clip with a fine-tuned Whisper checkpoint.

    Args:
        model_id: Hugging Face repository id, or a local checkpoint directory.
        clip_ids: identifiers to transcribe.
        audio_dir: directory holding one audio file per identifier.
        batch_size: clips per generate call.
        sample_rate: input sample rate; the extractor expects 16 kHz.

    Returns:
        Mapping of clip identifier to transcript.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # forced_decoder_ids is deprecated and conflicts with the modern generation config; leaving it
    # set makes generate emit a language token the fine-tune never trained against.
    model.generation_config.forced_decoder_ids = None

    predictions: dict[str, str] = {}
    for start in range(0, len(clip_ids), batch_size):
        batch = clip_ids[start : start + batch_size]
        waveforms = [load_clip(cid, audio_dir, sample_rate) for cid in batch]
        features = processor(
            waveforms, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            generated = model.generate(
                features, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS
            )
        for offset, clip_id in enumerate(batch):
            text = processor.batch_decode(generated[offset : offset + 1], skip_special_tokens=True)
            predictions[clip_id] = text[0].strip()
        if start % (batch_size * 20) == 0:
            print(f"[whisper] {min(start + batch_size, len(clip_ids))}/{len(clip_ids)}", flush=True)
    return predictions

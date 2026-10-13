# Tier 14 — Speech, Audio & Video AI — Revision Sheet

> Status: DRAFT. Generated from the concept files `14.01-audio-waveforms.md` … `14.33-audio-video-performance-and-cost-considerations.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 14.1 Audio waveforms | Pressure-versus-time stored as an array of samples; the rawest and longest audio representation. | *A waveform is a very long list of tiny pressure readings.* |
| 14.2 Sampling rate | Samples per second; must exceed twice the top frequency or aliasing folds it into lower frequencies. | *Sample at more than twice the highest frequency, or the high notes wear a disguise.* |
| 14.3 Amplitude and frequency | Amplitude = size (loudness), frequency = cycles per second (pitch); Fourier writes any sound as a sum of sines. | *Loudness is how tall, pitch is how fast; the FFT lists the ingredients.* |
| 14.4 Spectrograms | STFT magnitude as a time-frequency image; window length trades time vs frequency resolution. | *A spectrogram is the song's sheet music, drawn by an FFT sliding along the wave.* |
| 14.5 Mel spectrograms | STFT power pooled by perceptually spaced triangular filters and log-compressed: the standard speech feature. | *Mel features listen like an ear: sharp on the bass, blurry on the treble.* |
| 14.6 Audio preprocessing | Make every clip look like the training data: mono, resampled, normalised, padded, featurised. | *Garbage in, garbage out; make the audio look like what the model grew up on.* |
| 14.7 Audio embeddings | A learned vector summarising an audio clip, so similar sounds or speakers are close in space. | *An embedding is a fingerprint of meaning for sound.* |
| 14.8 Audio tokenization and neural codecs | Codec = encoder + residual vector quantizer + decoder, turning audio into a few integer tokens per frame. | *A codec is a zip file for sound whose words are the vocabulary of a speech language model.* |
| 14.9 Automatic Speech Recognition (ASR) | Neural speech-to-text; encoder over audio features with CTC, attention-decoder or transducer output. | *ASR is fill-in-the-words for a speech whose timing nobody told you.* |
| 14.10 CTC | Blank-augmented per-frame labels trained by summing over all alignments that collapse to the transcript. | *CTC: say a letter once, then let the pause tell you when a repeat is new.* |
| 14.11 Encoder-decoder ASR | Audio encoder + autoregressive text decoder with cross-attention; accurate but slower, can hallucinate. | *An encoder listens, a decoder writes, and cross-attention is the glance back at the audio.* |
| 14.12 Whisper as an ASR case study | Large-scale weakly supervised multilingual encoder-decoder ASR using 30 s log-mel windows (details REVIEW_REQUIRED). | *Whisper is a well-read generalist: great everywhere, flawless nowhere, and it will invent things if you hand it silence.* |
| 14.13 Speech segmentation | Detect speech regions and chunk long audio into clean segments, adding hangover to avoid clipping. | *Find the speech first, then transcribe; silence is not content.* |
| 14.14 Speaker diarization | Segment audio and cluster segments by voice to answer 'who spoke when' (anonymous labels). | *Diarization doesn't know your name, only that voice B spoke again.* |
| 14.15 Timestamps/alignment | Viterbi over a blank-extended transcript on frame posteriors gives per-word start and end times. | *Alignment is the highlighter that follows the singer.* |
| 14.16 Text-to-Speech (TTS) | Text to phonemes to acoustic features (mel or codec tokens) to waveform via a vocoder. | *TTS writes the score, the vocoder plays the instrument.* |
| 14.17 Vocoders | Turns mel/codec features into audio by reconstructing phase and detail; Griffin-Lim classically, GANs or diffusion today. | *A vocoder repaints the missing ripples.* |
| 14.18 Streaming speech | Incremental recognition with bounded delay: latency = chunk + lookahead + compute, traded against accuracy. | *Streaming is speed-reading with a short memory: lower delay, less context.* |
| 14.19 Realtime speech systems | Streaming ASR + LLM + streaming TTS with endpointing and barge-in, tuned for sub-second turn latency. | *Realtime speech is a relay race where the baton is a sentence and the clock never stops.* |
| 14.20 Audio evaluation | WER=(S+D+I)/N after consistent normalisation, plus slices, latency (RTF), and task-specific metrics like DER and MOS. | *WER counts the typos, but you still have to ask which words mattered.* |
| 14.21 Video as spatial + temporal data | A [T,H,W,C] tensor with spatial and temporal structure, huge and redundant. | *Video = images x time, and almost all of it is repeats.* |
| 14.22 Video frames | One image per time step; compressed streams store keyframes plus deltas with timestamps. | *Frames are the pages; keyframes are the pages stored in full.* |
| 14.23 Frame sampling | Choose frames by uniform-K, stride, target fps or adaptively, trading coverage, temporal resolution and cost. | *Sample like a photographer: enough shots to tell the story, not every blink.* |
| 14.24 Scene/shot segmentation | Detect spikes in frame-to-frame difference to find cuts; group shots into scenes by similarity. | *A cut is a spike in how different this frame is from the last.* |
| 14.25 Temporal representation | Spatio-temporal tokens (tubelets, 3-D conv, flow) with time position encodings. | *Time is a third axis; tubelets patch across it.* |
| 14.26 Temporal attention | Self-attention along time with full, local, causal or factorized patterns to control quadratic cost. | *First look around, then look across time; never both at once.* |
| 14.27 Video embeddings | Pooled vector of encoded frames/tubelets summarising a clip for search and downstream models. | *A video embedding is an index card for a moving picture.* |
| 14.28 Video-language models | Sampled frames encoded and projected into an LLM's token space; limited by token budget. | *A video LLM is a fast-reading friend with a limited photo budget.* |
| 14.29 Audio-video-text alignment | Map words, shots and speakers onto one timeline (and shared embedding space) so retrieval can cite moments. | *Alignment is the timecode that turns three modalities into one story.* |
| 14.30 Multimodal synchronization | Estimate and correct constant offset and drift so audio and video share one timeline. | *Sync is making the lips and the voice agree on what time it is.* |
| 14.31 Streaming/realtime video considerations | Bounded-latency, causal, sparsely sampled video processing with drop-to-latest buffering. | *In live video, a stale frame is worse than a dropped one.* |
| 14.32 Video evaluation | Task metrics (acc, recall@K, IoU, F1) plus controls proving the model uses time, reported per slice. | *Test whether the model watched the video or just read the title.* |
| 14.33 Audio/video performance and cost considerations | Duration x frames x tokens x model cost: cut volume (VAD, sampling), cache, tier models, and compare API vs self-host. | *Audio and video are heavy; process less, cache more, measure everything.* |

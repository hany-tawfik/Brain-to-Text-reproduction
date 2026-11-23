# Background primer

The concepts behind this project, mapped to EEG and audio ML, which is where I come from.

## Recording: Utah arrays vs EEG

| | EEG | Intracortical arrays (this data) |
|---|---|---|
| Sensor | Scalp electrodes | 4 arrays x 64 microelectrodes, inside motor cortex |
| Signal | Summed field potentials from millions of neurons | Spikes from small groups of neurons near each electrode |
| Spatial resolution | cm | ~0.4 mm electrode spacing |
| SNR for speech | Low, research stage | High enough for ~95% word accuracy in the best systems |

## The two features per electrode

- **Threshold crossings:** count of times the high-pass filtered voltage drops below -4.5 x RMS in each 20 ms bin. A rough spike count without sorting individual neurons.
- **Spike band power:** power in the spike frequency band (roughly 250 to 5000 Hz) per bin. Think of it as a band-energy feature, like one band of a filterbank.

So each trial is a matrix of time (20 ms frames) x 512 features. That is structurally the same as a log-mel spectrogram: frames x bins. Most ASR tools apply directly.

## Brain areas in the data

- **6v (ventral premotor), split into ventral and dorsal arrays:** strongest speech signal. Some papers use only the 128 channels in 6v
- **Area 4:** primary motor cortex
- **55b:** a speech-related area near the premotor cortex

## Day-to-day drift

Electrode signals shift between sessions (tissue response, micromotion). Same problem as EEG session effects. The baseline handles it with a separate input layer per day (45 small linear layers, then a shared GRU). Performance tends to drop for later sessions. See the per-session figure in `results/`.

## Decoding pipeline

1. **Neural → phoneme probabilities.** A GRU outputs, for every frame, a probability over 41 classes: 39 English phonemes (CMU set), a word-boundary/silence token, and the CTC blank.
2. **CTC (Connectionist Temporal Classification).** Training loss that lets the model learn alignment between frames and a shorter phoneme sequence without frame-level labels. Same loss used in classic ASR (DeepSpeech, wav2vec 2.0 fine-tuning).
   - Greedy decoding: take the argmax per frame, collapse repeats, drop blanks.
3. **Phonemes → words.** An n-gram language model in a WFST decoder turns phoneme probabilities into candidate sentences.
4. **LLM rescoring.** OPT 6.7B re-ranks the top candidates. B2T '24 winners used fine-tuned LLMs to merge outputs from several decoders.

## Metrics

- **PER (phoneme error rate):** edit distance between predicted and true phoneme sequences, divided by true length. Depends only on the neural model. Measurable on a Mac without the language model.
- **WER (word error rate):** same, at the word level, after the language model. This is the competition metric.
- PER improvements do not always carry over to WER.

## Patch embedding

The baseline groups 14 frames with stride 4 into one input step (like framing audio before an encoder). This shortens sequences 4x and speeds up the GRU.

## Where audio ML experience helps

- Treating neural features as a spectrogram: SpecAugment, normalization, filterbank thinking
- ASR encoders (Conformer), CTC, beam search, LM fusion
- Pretrained speech models as decoders (Whisper-style)
- Real-time constraints: the clinical system decodes online, like a live audio pipeline

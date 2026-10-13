# Tier 5 — author open items

## All 27 concepts (5.1–5.27) and the build

- TESTED: every runnable example on the 27 pages is executed by `tests/test_md_snippets.py`; `code/seqmodels.py` has 10 tests (RNN/LSTM/GRU cells equal `torch.nn` to 1e-6, manual BPTT equals finite differences, signal decay, LSTM memory beats RNN, attention fixes long reversal, attention rows sum to 1 and align on the mirror position, word2vec skip-gram and CBOW cluster same-role words) and `lab/` 7. Environment: Python 3.14, PyTorch 2.14.1, NumPy 2.5.4, CPU only.
- Honest results kept in the text: all three recurrent char models tie on short-range text (5.10 lab); teacher-forcing ratio barely matters on the easy reversal task (5.22); the GRU fails the memory task earlier than a forget-bias-initialized LSTM (5.14 figure); an LSTM with default bias also loses the gradient over 100 steps (5.13 ③: forget bias 1 is what keeps it); the memory-task break points are single-seed.
- Fixed while authoring: CBOW needed a larger learning rate than skip-gram (documented in the page and library); the PyTorch word2vec page only clustered correctly after zero-initializing the output vectors (as the original does); a first PyTorch GloVe attempt overfitted with default N(0,1) initialization; the BPTT test originally measured gradient flow to all steps from every step's loss instead of from the last step only.

## NOT_TESTED / REVIEW_REQUIRED

- Attributions and facts to verify with sources: Mikolov et al. 2013 (Word2Vec) and the Google News figures; Pennington et al. 2014 (GloVe) and `x_max`, `α`; Bojanowski et al. 2017 (FastText); Hochreiter & Schmidhuber 1997, Gers et al. 2000 (forget gate), Cho et al. 2014 (GRU), Chung et al. 2014, Greff et al. 2017, Jozefowicz et al. 2015 (forget bias); Bengio et al. 1994, Pascanu et al. 2013; Sutskever et al. 2014; Bahdanau et al. 2014; Luong et al. 2015; Bengio et al. 2015 and Ranzato et al. 2016 (scheduled sampling, exposure bias); Kaplan et al. 2020; Vaswani et al. 2017; GNMT and Whisper descriptions; interpretable character-RNN units.
- Simplifications: the tier's Luong model is dot-product attention without input feeding or the extra `tanh` attentional layer; the Bahdanau model is unidirectional (the paper used a bidirectional encoder); the 5.17 gate averages come from a 1,000-step toy model.
- 5.14, 5.19, 5.20 train small PyTorch models inside the page examples; their exact accuracies (all 1.00 or at chance) are stable on the development machine but are the least portable lines on the pages (the snippet check allows small numeric differences only).
- DUPLICATION / ORDER: 5.4–5.9 overlap Tier 6 (tokenization and embeddings, drafted separately on `chatgpt-bgc-1012`); 5.12–5.13 overlap 3.20 and 3.25–3.26; 5.27 previews Tiers 7–8. Pages cross-reference rather than repeat.
- Open design question: the roadmap lists word embeddings (5.4–5.9) in Tier 5 and again in Tier 6; the owner may want one of them to be authoritative.

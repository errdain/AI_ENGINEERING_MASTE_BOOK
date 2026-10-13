AI Engineering Master Notebook

Roadmap V1.0 --- FROZEN Curriculum

Goal: Build enough understanding that you can explain → implement →
debug → compare → architect → optimize → secure → deploy each major
AI-engineering concept.

The book will follow seven phases: PHASE 1 Deep Learning Foundations ↓
PHASE 2 Transformers & Foundation Models ↓ PHASE 3 LLM Application
Engineering ↓ PHASE 4 RAG & Multimodal Systems ↓ PHASE 5 Agents &
Interoperability ↓ PHASE 6 Enterprise AI Engineering ↓ PHASE 7
Architecture & Enterprise Projects

PHASE 1 --- DEEP LEARNING FOUNDATIONS

Tier 1 --- Mathematics for AI

1.1 Scalars

1.2 Vectors

1.3 Matrices

1.4 Tensors

1.5 Tensor shapes

1.6 Dimensions and axes

1.7 Broadcasting

1.8 Vector addition

1.9 Scalar multiplication

1.10 Dot product

1.11 Matrix multiplication

1.12 Matrix transpose

1.13 Identity matrices

1.14 Matrix inverse

1.15 Norms

1.16 Distance metrics

1.17 Cosine similarity

1.18 Eigenvalues/eigenvectors

1.19 Singular Value Decomposition

1.20 Functions

1.21 Derivatives

1.22 Partial derivatives

1.23 Gradients

1.24 Jacobians

1.25 Chain rule

1.26 Probability

1.27 Random variables

1.28 Probability distributions

1.29 Mean/expectation

1.30 Variance

1.31 Standard deviation

1.32 Covariance

1.33 Conditional probability

1.34 Bayes' theorem

1.35 Maximum likelihood

1.36 Entropy

1.37 Cross-entropy

1.38 KL divergence

1.39 Numerical stability

1.40 Floating-point precision

1.41 Logarithms

1.42 Exponentials

1.43 Log-sum-exp

1.44 Numerical overflow/underflow

1.45 Convexity intuition

1.46 Optimization landscapes

1.47 Local/global minima intuition

Build: NumPy/PyTorch Mathematics Lab.

⸻

Tier 2 --- Machine Learning Fundamentals

2.1 Artificial Intelligence vs ML vs DL

2.2 Supervised learning

2.3 Unsupervised learning

2.4 Semi-supervised learning

2.5 Self-supervised learning

2.6 Reinforcement learning overview

2.7 Features

2.8 Labels

2.9 Samples

2.10 Dataset construction

2.11 Training dataset

2.12 Validation dataset

2.13 Test dataset

2.14 Classification

2.15 Regression

2.16 Clustering

2.17 Loss functions

2.18 Objective functions

2.19 Gradient descent

2.20 Stochastic gradient descent

2.21 Mini-batch training

2.22 Epoch

2.23 Batch

2.24 Iteration

2.25 Learning rate

2.26 Underfitting

2.27 Overfitting

2.28 Bias

2.29 Variance

2.30 Bias--variance trade-off

2.31 Generalization

2.32 Regularization

2.33 Data leakage

2.34 Distribution shift

2.35 Hyperparameters

2.36 Hyperparameter tuning

2.37 Accuracy

2.38 Precision

2.39 Recall

2.40 F1

2.41 ROC/AUC

2.42 Confusion matrix

2.43 Calibration

2.44 Cross-validation

2.45 Class imbalance

2.46 Baseline models

2.47 Feature scaling/normalization

2.48 Feature engineering overview

2.49 Data preprocessing

2.50 Dataset imbalance

2.51 Metric selection

Build: small ML training/evaluation laboratory.

⸻

Tier 3 --- Neural Networks

3.1 Biological inspiration vs artificial neuron

3.2 Perceptron

3.3 Input layer

3.4 Hidden layers

3.5 Output layer

3.6 Weights

3.7 Bias

3.8 Weighted sum

3.9 Activation functions

3.10 Sigmoid

3.11 Tanh

3.12 ReLU

3.13 GELU

3.14 SiLU/Swish

3.15 Softmax

3.16 Linear/Dense layer

3.17 MLP

3.18 Forward propagation

3.19 Computational graph

3.20 Backpropagation

3.21 Automatic differentiation

3.22 Parameter initialization

3.23 Xavier initialization

3.24 He initialization

3.25 Vanishing gradients

3.26 Exploding gradients

3.27 Residual connections

3.28 Dropout

3.29 Weight decay

3.30 Batch normalization

3.31 Layer normalization

3.32 RMSNorm

Build: neural network first manually, then in PyTorch.

⸻

Tier 4 --- Deep Learning Training & Optimization

4.1 Training loop

4.2 Forward pass

4.3 Loss calculation

4.4 Backward pass

4.5 Parameter update

4.6 SGD

4.7 Momentum

4.8 Adam

4.9 AdamW

4.10 Learning-rate scheduling

4.11 Warmup

4.12 Cosine decay

4.13 Batch size

4.14 Gradient accumulation

4.15 Gradient clipping

4.16 FP32

4.17 FP16

4.18 BF16

4.19 Mixed-precision training

4.20 Checkpoints

4.21 Reproducibility

4.22 Seeds

4.23 CPU vs GPU

4.24 GPU memory

4.25 Profiling

4.26 DataLoader

4.27 PyTorch tensors

4.28 nn.Module

4.29 Optimizers

4.30 Distributed-training introduction

4.31 Data parallelism

4.32 Model/tensor parallelism

4.33 Pipeline parallelism

4.34 FSDP/ZeRO overview

4.35 Training curves

4.36 Validation curves

4.37 Gradient monitoring

4.38 NaN/Inf debugging

4.39 Loss debugging

4.40 Memory profiling

4.41 Gradient checkpointing

Build: complete PyTorch training pipeline + profiler.

⸻

Tier 5 --- Sequence Models

5.1 Sequential data

5.2 Language as sequence

5.3 One-hot encoding

5.4 Word embeddings

5.5 Word2Vec

5.6 CBOW

5.7 Skip-gram

5.8 GloVe

5.9 FastText

5.10 RNN

5.11 Hidden state

5.12 BPTT

5.13 RNN gradient problems

5.14 LSTM

5.15 Cell state

5.16 Forget gate

5.17 Input gate

5.18 Output gate

5.19 GRU

5.20 Encoder-decoder

5.21 Seq2Seq

5.22 Teacher forcing

5.23 Exposure bias

5.24 Bahdanau attention

5.25 Luong attention

5.26 Why RNNs became a bottleneck

5.27 Transition to Transformers

Build: character RNN → LSTM → attention Seq2Seq comparison.

⸻

PHASE 2 --- TRANSFORMERS & FOUNDATION MODELS

Tier 6 --- Tokenization & Embeddings

6.1 Text preprocessing

6.2 Character tokenization

6.3 Word tokenization

6.4 Byte tokenization

6.5 Subword tokenization

6.6 Vocabulary

6.7 Token IDs

6.8 UNK

6.9 PAD

6.10 BOS

6.11 EOS

6.12 BPE

6.13 WordPiece

6.14 SentencePiece

6.15 Unigram LM tokenizer

6.16 Attention masks

6.17 Token embedding

6.18 Embedding matrix

6.19 Embedding dimensions

6.20 Positional information

6.21 Learned positional embeddings

6.22 Sinusoidal positional encoding

6.23 Relative positional encoding

6.24 RoPE

6.25 ALiBi

6.26 Token vs semantic embeddings

6.27 Tokenizer training 6.28 Vocabulary-size trade-offs 6.29 Byte
fallback 6.30 Unicode normalization 6.31 Tokenization efficiency 6.32
Tokens-per-character / compression intuition

Build: miniature BPE tokenizer.

⸻

Tier 7 --- Attention

7.1 Why attention

7.2 Query

7.3 Key

7.4 Value

7.5 Q/K/V projections

7.6 Dot-product similarity

7.7 Attention scores

7.8 Scaling by √dk

7.9 Softmax

7.10 Weighted values

7.11 Scaled dot-product attention

7.12 Self-attention

7.13 Cross-attention

7.14 Causal attention

7.15 Bidirectional attention

7.16 Attention masks

7.17 Padding masks

7.18 Multi-head attention

7.19 Head dimensions

7.20 Projection matrices

7.21 MQA

7.22 GQA

7.23 Attention complexity

7.24 Attention memory requirements

7.25 FlashAttention

7.26 Attention visualization

7.27 Attention patterns

7.28 Attention sinks

7.29 Long-context attention limitations

7.30 Sliding-window attention --- introduction only; full architecture
later.

Build: attention from matrix operations → MHA.

⸻

Tier 8 --- Transformer Architecture

8.1 Original Transformer

8.2 Transformer encoder

8.3 Transformer decoder

8.4 Encoder-decoder architecture

8.5 Transformer block

8.6 Self-attention layer

8.7 Cross-attention layer

8.8 Feed-forward network

8.9 Position-wise MLP

8.10 Residual connections

8.11 LayerNorm

8.12 Pre-LN

8.13 Post-LN

8.14 RMSNorm

8.15 GELU

8.16 SwiGLU

8.17 Layer stacking

8.18 Hidden dimension

8.19 Number of heads

8.20 Number of layers

8.21 Parameter counting

8.22 Weight tying

8.23 Encoder-only architecture

8.24 BERT

8.25 Decoder-only architecture

8.26 GPT

8.27 Encoder-decoder architecture

8.28 T5/BART

8.29 Causal objective

8.30 Masked objective

Major build: Mini-GPT.

Tier 9 --- Foundation Model Pretraining

9.1 Pretraining overview 9.2 Pretraining data acquisition 9.3 Data
cleaning 9.4 Quality filtering 9.5 Deduplication strategies 9.6 Data
mixture design 9.7 Language balancing 9.8 Domain balancing 9.9 Data
provenance 9.10 Data licensing 9.11 Data contamination 9.12 Benchmark
contamination 9.13 Data curriculum 9.14 Token budgets 9.15 Compute
budgets 9.16 Self-supervised learning for foundation models 9.17
Next-token prediction 9.18 Masked-language modeling 9.19 Pretraining
objectives 9.20 Scaling laws 9.21 Compute/data/model scaling trade-offs
9.22 Training checkpoints 9.23 Distributed pretraining 9.24 Dense-model
pretraining 9.25 Mixture-of-Experts pretraining 9.26 Expert routing 9.27
Load balancing 9.28 Training stability 9.29 Continued pretraining 9.30
Domain-adaptive pretraining 9.31 Pretraining evaluation

Build: small-scale language-model pretraining laboratory with dataset
preparation, checkpoints, metrics and reproducibility.

⸻

Tier 10 --- Fine-Tuning & Alignment

10.1 Base models vs instruction-tuned models 10.2 Post-training overview
10.3 Supervised fine-tuning (SFT) 10.4 Instruction datasets 10.5 Chat
templates 10.6 SFT data curation 10.7 Synthetic training data 10.8
Reasoning-data construction 10.9 Full fine-tuning 10.10
Parameter-efficient fine-tuning (PEFT) 10.11 LoRA 10.12 LoRA rank 10.13
LoRA alpha 10.14 LoRA target modules 10.15 QLoRA 10.16 Adapters 10.17
Prompt tuning 10.18 Quantization-aware fine-tuning concepts 10.19
Preference data 10.20 Preference optimization 10.21 Reward models 10.22
RLHF 10.23 PPO intuition for alignment 10.24 On-policy vs off-policy
alignment intuition 10.25 Direct Preference Optimization (DPO) 10.26
Distillation 10.27 Continued fine-tuning 10.28 Catastrophic forgetting
10.29 Catastrophic-forgetting evaluation 10.30 Adapter merging 10.31
Model merging 10.32 Checkpoint merging 10.33 Fine-tuning evaluation
10.34 Fine-tuning data/security considerations 10.35 Fine-tuning cost
and deployment trade-offs

Build: compare full fine-tuning concepts, LoRA/QLoRA and
preference-optimization workflows on a small model where practical.

⸻

Tier 11 --- LLM Inference & Optimization

11.1 Autoregressive inference 11.2 Logits 11.3 Decoding overview 11.4
Greedy decoding 11.5 Temperature 11.6 Top-k sampling 11.7 Top-p sampling
11.8 Repetition/frequency penalties 11.9 Beam search 11.10 Stop
sequences 11.11 Prefill phase 11.12 Decode phase 11.13 Time to first
token (TTFT) 11.14 Inter-token latency 11.15 Tokens per second 11.16 KV
cache 11.17 KV-cache memory calculation 11.18 Prefix caching 11.19
Batching 11.20 Continuous batching 11.21 PagedAttention 11.22 Request
scheduling 11.23 Backpressure 11.24 Admission control 11.25 Speculative
decoding 11.26 Quantization 11.27 INT8 inference 11.28 INT4 inference
11.29 GGUF 11.30 llama.cpp 11.31 Ollama 11.32 vLLM 11.33 Local vs hosted
inference 11.34 Model replicas 11.35 Tensor-parallel inference 11.36
Pipeline-parallel inference 11.37 Serving autoscaling 11.38 Cold starts
11.39 Memory bandwidth 11.40 Compute-bound vs memory-bound inference
11.41 Prefill vs decode bottlenecks 11.42 Concurrency 11.43 Throughput
11.44 GPU-memory budgeting 11.45 Inference cost 11.46 Serving
reliability 11.47 Inference observability

Build: inference benchmark comparing decoding, caching, quantization and
serving configurations.

⸻

Tier 12 --- Modern Foundation-Model Architectures

12.1 Architectural evolution after the original Transformer 12.2 Dense
Transformers 12.3 Encoder-only models as an architectural family 12.4
Decoder-only models as an architectural family 12.5 Encoder-decoder
models as an architectural family 12.6 Multi-Query Attention (MQA) in
modern models 12.7 Grouped-Query Attention (GQA) in modern models 12.8
RoPE variants and extensions 12.9 Long-context architectures 12.10
Sliding-window attention 12.11 Sparse attention 12.12
Efficient-attention architecture patterns 12.13 Mixture of Experts (MoE)
12.14 Sparse MoE 12.15 Expert routing 12.16 Expert load balancing 12.17
State Space Models (SSMs) 12.18 Mamba 12.19 Transformer-SSM hybrid
architectures 12.20 Multimodal foundation-model architectures 12.21
Retrieval-enhanced model architectures 12.22 Reasoning-oriented
training/inference patterns 12.23 Architecture selection trade-offs
12.24 Architecture case study: BERT family 12.25 Architecture case
study: GPT family 12.26 Architecture case study: T5/BART family 12.27
Architecture case study: Llama family 12.28 Architecture case study:
Qwen family 12.29 Architecture case study: Gemma family 12.30 Comparing
modern model families without depending on vendor-specific APIs

Rule: named model families are case studies; durable architectural ideas
are authoritative.

Build: architecture comparison matrix covering quality, context, memory,
inference and deployment trade-offs.

⸻

Tier 13 --- Vision AI

13.1 Digital image representation 13.2 Pixels, channels and image
tensors 13.3 Image preprocessing 13.4 Convolutional Neural Networks
(CNNs) 13.5 Convolution 13.6 Kernels/filters 13.7 Stride 13.8 Padding
13.9 Pooling 13.10 Receptive fields 13.11 Feature maps 13.12 Residual
vision networks 13.13 ResNet 13.14 Image embeddings 13.15 Vision
Transformer (ViT) 13.16 Image patches 13.17 Patch embeddings 13.18
Vision encoders 13.19 Contrastive learning 13.20 CLIP 13.21 Shared
vision-language embedding spaces 13.22 Vision-language models (VLMs)
13.23 Cross-modal attention concepts 13.24 Visual instruction tuning
13.25 OCR fundamentals 13.26 Object detection 13.27 Image segmentation
13.28 Document-image understanding 13.29 Chart/table visual
understanding 13.30 Diffusion-model introduction 13.31 Vision evaluation
13.32 Vision deployment/performance considerations

Build: image-embedding, classification/retrieval and ViT/VLM comparison
laboratory.

⸻

Tier 14 --- Speech, Audio & Video AI

14.1 Audio waveforms 14.2 Sampling rate 14.3 Amplitude and frequency
14.4 Spectrograms 14.5 Mel spectrograms 14.6 Audio preprocessing 14.7
Audio embeddings 14.8 Audio tokenization and neural codecs 14.9
Automatic Speech Recognition (ASR) 14.10 CTC 14.11 Encoder-decoder ASR
14.12 Whisper as an ASR case study 14.13 Speech segmentation 14.14
Speaker diarization 14.15 Timestamps/alignment 14.16 Text-to-Speech
(TTS) 14.17 Vocoders 14.18 Streaming speech 14.19 Realtime speech
systems 14.20 Audio evaluation 14.21 Video as spatial + temporal data
14.22 Video frames 14.23 Frame sampling 14.24 Scene/shot segmentation
14.25 Temporal representation 14.26 Temporal attention 14.27 Video
embeddings 14.28 Video-language models 14.29 Audio-video-text alignment
14.30 Multimodal synchronization 14.31 Streaming/realtime video
considerations 14.32 Video evaluation 14.33 Audio/video performance and
cost considerations

Build: speech transcription + audio/video representation laboratory with
latency and quality measurements.

⸻

Tier 15 --- Foundation-Model Evaluation & Safety

15.1 Why foundation-model evaluation is difficult 15.2 Evaluation
objectives 15.3 Offline vs online evaluation 15.4 Intrinsic vs extrinsic
evaluation 15.5 Perplexity 15.6 Benchmark design 15.7 Benchmark datasets
15.8 Task-specific evaluation 15.9 Accuracy 15.10 Precision/recall/F1 in
model evaluation 15.11 Calibration 15.12 Factuality 15.13 Hallucination
15.14 Groundedness 15.15 Robustness 15.16 Adversarial testing 15.17
Distribution-shift evaluation 15.18 Bias evaluation 15.19 Fairness
considerations 15.20 Privacy evaluation 15.21 Memorization/data-leakage
evaluation 15.22 Model safety 15.23 Harmful-output evaluation 15.24 Red
teaming 15.25 Regression datasets 15.26 Golden datasets 15.27 Human
evaluation 15.28 LLM-as-judge concepts and limitations 15.29 Evaluation
reproducibility 15.30 Evaluation versioning 15.31 Evaluation pipelines
15.32 Evaluation gates 15.33 Model comparison 15.34 Release/readiness
criteria 15.35 Safety vs capability trade-offs

Build: reusable model-evaluation harness with regression datasets, human
review and release gates.

PHASE 3 --- LLM APPLICATION ENGINEERING

Tier 16 --- Raw LLM Engineering & Multi-Provider Systems

Priority: ★★★★★ · CORE · CODE · INT · ARCH · PROD

Provider fundamentals

16.1 What is an LLM API? 16.2 REST API fundamentals 16.3 SDK vs REST
16.4 HTTP request/response lifecycle 16.5 Synchronous vs asynchronous
calls 16.6 Local vs hosted LLMs 16.7 Proprietary vs open-weight models
16.8 Model providers 16.9 Model capability discovery 16.10
Provider-specific parameters

Providers

16.11 Ollama 16.12 OpenAI API concepts 16.13 Anthropic API concepts
16.14 Gemini API concepts 16.15 Hugging Face inference concepts 16.16
Local inference servers 16.17 Provider abstraction 16.18 Adapter pattern
16.19 Model gateway 16.20 Unified request schema 16.21 Unified response
schema

Conversation

16.22 System message 16.23 User message 16.24 Assistant message 16.25
Conversation history 16.26 Chat templates 16.27 Context window 16.28
Input tokens 16.29 Output tokens 16.30 Token budgets

Generation

16.31 Temperature 16.32 Top-p 16.33 Top-k 16.34 Maximum output tokens
16.35 Stop sequences 16.36 Deterministic vs stochastic generation

Streaming

16.37 Streaming generation 16.38 Token streaming 16.39 Server-Sent
Events 16.40 WebSockets 16.41 Stream cancellation 16.42 Backpressure

Reliability

16.43 Timeout handling 16.44 Authentication errors 16.45 Rate limits
16.46 Connection errors 16.47 Provider errors 16.48 Retry strategies
16.49 Exponential backoff 16.50 Jitter 16.51 Provider fallback 16.52
Model fallback 16.53 Circuit breaker concept

Performance

16.54 End-to-end latency 16.55 Time to first token 16.56 Tokens per
second 16.57 Throughput 16.58 Concurrency 16.59 Token usage 16.60 Cost
calculation

Security

16.61 API key management 16.62 Environment variables 16.63 Secret
managers 16.64 Key masking 16.65 Logging without exposing secrets

Observability foundations

16.66 Request IDs 16.67 Correlation IDs 16.68 Structured logging 16.69
Request metadata 16.70 Basic tracing

Build

AI Engineering Playground V1 --- Raw Python Multi-Provider LLM
Playground

No LangChain yet.

⸻

Tier 17 --- Prompt & Context Engineering

CORE · INT · CODE

17.1 What is a prompt? 17.2 Prompt engineering vs context engineering
17.3 System instructions 17.4 User instructions 17.5 Instruction
hierarchy 17.6 Zero-shot prompting 17.7 One-shot prompting 17.8 Few-shot
prompting 17.9 In-context learning 17.10 Role prompting 17.11
Constraints 17.12 Delimiters 17.13 Examples 17.14 Output-format
instructions 17.15 Prompt templates 17.16 Template variables 17.17
Dynamic prompts 17.18 Prompt composition 17.19 Task decomposition 17.20
Structured reasoning workflows 17.21 Context selection 17.22 Context
ordering 17.23 Context truncation 17.24 Context compression 17.25
Context summarization 17.26 Lost-in-the-middle problem 17.27 Token
budgeting 17.28 Long-context strategies 17.29 Prompt caching 17.30
Semantic caching 17.31 Prompt versioning 17.32 Prompt A/B testing 17.33
Prompt regression testing 17.34 Prompt injection fundamentals 17.35
Untrusted context 17.36 Grounding 17.37 Citations 17.38 Abstention 17.39
Hallucination reduction 17.40 Prompt evaluation

Build

Prompt Engineering Laboratory

Compare: Prompt ↓ Model A ──→ Response ↓ Model B ──→ Response ↓ Model C
──→ Response ↓ Quality + Tokens + Latency + Cost

Tier 18 --- Memory Systems

CORE · CODE · ARCH · INT

18.1 Stateless applications 18.2 Stateful applications 18.3 Conversation
state 18.4 Short-term memory 18.5 Conversation buffer 18.6
Sliding-window memory 18.7 Token-window memory 18.8 Summary memory 18.9
Hierarchical summaries 18.10 Long-term memory 18.11 Semantic memory
18.12 Episodic memory 18.13 Procedural memory 18.14 User-profile memory
18.15 Preference memory 18.16 Vector-based memory 18.17 Memory
extraction 18.18 Memory consolidation 18.19 Memory retrieval 18.20
Memory ranking 18.21 Memory relevance 18.22 Memory expiration 18.23
Memory deletion 18.24 Memory correction 18.25 Memory contamination 18.26
Cross-session memory 18.27 Persistent conversation storage 18.28
Database-backed memory 18.29 Memory privacy 18.30 Memory authorization
18.31 Memory evaluation 18.32 Memory vs RAG

Build

Playground V2 --- selectable memory architectures.

⸻

Tier 19 --- Structured Outputs & Tool Calling

★★★★★ CORE

Structured generation

19.1 Structured outputs 19.2 JSON output 19.3 JSON Schema 19.4 Pydantic
19.5 Schema-constrained generation 19.6 Validation 19.7 Parsing failures
19.8 Retry/repair strategies

Tool calling

19.9 What is a tool? 19.10 Function calling 19.11 Tool definitions 19.12
Tool descriptions 19.13 Tool schemas 19.14 Tool selection 19.15 Argument
generation 19.16 Argument validation 19.17 Tool execution 19.18 Tool
results 19.19 Tool execution loop 19.20 Sequential tools 19.21 Parallel
tools 19.22 Tool chaining 19.23 Tool timeouts 19.24 Tool retries 19.25
Idempotency 19.26 Tool failure recovery

Tool categories

19.27 Calculator tools 19.28 Database tools 19.29 Search tools 19.30 API
tools 19.31 File tools 19.32 Code-execution tools

Security

19.33 Tool permissions 19.34 Least privilege 19.35 Tool authorization
19.36 Tool isolation 19.37 Sandboxing 19.38 Human approval 19.39
Consequential actions 19.40 Audit logs

Evaluation

19.41 Tool-selection accuracy 19.42 Argument accuracy 19.43 Execution
success 19.44 Tool-call latency 19.45 Tool-call tracing

Build

Playground V3 User ↓ LLM ↓ Tool selection ↓ Schema validation ↓
Authorization ↓ Tool execution ↓ Tool result ↓ LLM ↓ Answer

This tier is where your playground begins transitioning from an LLM
application into an AI system.

⸻

PHASE 4 --- RAG & MULTIMODAL AI

Tier 20 --- RAG Fundamentals

★★★★★ CORE · CODE · INT · ARCH

20.1 What is RAG? 20.2 Why RAG exists 20.3 Parametric vs external
knowledge 20.4 RAG vs fine-tuning 20.5 RAG vs long context 20.6 RAG
architecture 20.7 Knowledge sources 20.8 Connectors 20.9 Document
ingestion 20.10 PDF parsing 20.11 HTML parsing 20.12 Markdown parsing
20.13 Structured-data ingestion 20.14 Cleaning 20.15 Normalization 20.16
Chunking 20.17 Fixed-size chunking 20.18 Recursive chunking 20.19
Semantic chunking 20.20 Structure-aware chunking 20.21 Chunk size 20.22
Chunk overlap 20.23 Metadata 20.24 Embeddings 20.25 Embedding models
20.26 Dense vectors 20.27 Similarity search 20.28 Cosine similarity
20.29 Dot-product similarity 20.30 Euclidean distance 20.31 Vector
databases 20.32 ANN 20.33 HNSW 20.34 IVF concepts 20.35 FAISS 20.36
Chroma 20.37 Qdrant 20.38 pgvector 20.39 Managed vector databases 20.40
Indexing 20.41 Retrieval 20.42 Top-K 20.43 Similarity threshold 20.44
Context construction 20.45 RAG prompts 20.46 Generation 20.47 Grounding
20.48 Citations 20.49 Attribution 20.50 Basic RAG evaluation 20.51
Retrieval failures 20.52 Generation failures

Build

RAG V1 from scratch.

⸻

Tier 21 --- Advanced RAG

★★★★★

21.1 Sparse retrieval 21.2 TF-IDF 21.3 BM25 21.4 Dense retrieval 21.5
Sparse vs dense 21.6 Hybrid retrieval 21.7 Reciprocal Rank Fusion 21.8
Metadata filtering 21.9 Access-aware retrieval 21.10 Query rewriting
21.11 Query expansion 21.12 Query decomposition 21.13 Multi-query
retrieval 21.14 HyDE 21.15 Reranking 21.16 Cross-encoder reranking 21.17
Bi-encoder vs cross-encoder 21.18 Parent-child retrieval 21.19
Hierarchical retrieval 21.20 Multi-vector retrieval 21.21 Late
interaction 21.22 Contextual retrieval 21.23 Contextual compression
21.24 Adaptive retrieval 21.25 Corrective RAG 21.26 Self-RAG concepts
21.27 Retrieval caching 21.28 Embedding caching 21.29 Index refresh
21.30 Incremental indexing 21.31 Retrieval latency 21.32 Retrieval cost
21.33 Retrieval quality 21.34 Recall@K 21.35 Precision@K 21.36 MRR 21.37
NDCG

Build

RAG V2 comparison laboratory.

⸻

Tier 22 --- GraphRAG & Agentic RAG

Graph foundations

22.1 Knowledge graphs 22.2 Nodes 22.3 Edges 22.4 Properties 22.5
Entities 22.6 Relationships 22.7 Entity extraction 22.8 Relation
extraction 22.9 Entity resolution 22.10 Graph construction 22.11 Graph
databases 22.12 Cypher concepts 22.13 Graph traversal 22.14 Graph
retrieval

GraphRAG

22.15 GraphRAG concept 22.16 Graph indexing 22.17 Communities 22.18
Community summaries 22.19 Local search 22.20 Global search 22.21
Multi-hop retrieval

Agentic retrieval

22.22 Retrieval planning 22.23 Dynamic source selection 22.24 Iterative
retrieval 22.25 Evidence verification 22.26 Retrieval reflection 22.27
SQL retrieval 22.28 Text-to-SQL 22.29 Vector + SQL 22.30 Vector + graph
22.31 SQL + graph 22.32 Vector + SQL + graph 22.33 Cross-source
synthesis

Build

Enterprise Hybrid Knowledge Assistant.

⸻

Tier 23 --- Multimodal AI & Multimodal RAG

Modalities

23.1 Modality 23.2 Text 23.3 Images 23.4 Audio 23.5 Video 23.6 Documents
23.7 Tables 23.8 Charts

Multimodal representation

23.9 Modality encoders 23.10 Text encoders 23.11 Vision encoders 23.12
Audio encoders 23.13 Shared embedding spaces 23.14 Cross-modal alignment
23.15 Fusion strategies

Vision

23.16 Image understanding 23.17 OCR 23.18 Document layout 23.19 Tables
23.20 Charts 23.21 Screenshots 23.22 Page grounding

Audio

23.23 Speech-to-text 23.24 Speaker diarization 23.25 Timestamps 23.26
Text-to-speech 23.27 Streaming audio 23.28 Realtime conversation

Video

23.29 Frame extraction 23.30 Temporal sampling 23.31 Scene segmentation
23.32 Video embeddings 23.33 Temporal retrieval

Multimodal RAG

23.34 Multimodal ingestion 23.35 Multimodal chunking 23.36 Image-text
retrieval 23.37 Audio-text retrieval 23.38 Video-text retrieval 23.39
Multimodal indexing 23.40 Multimodal retrieval 23.41 Evidence alignment
23.42 VLM generation 23.43 Multimodal citations 23.44 Multimodal
evaluation

Build

PDF ──────┐ Image ────┤ Audio ────┤ Video ────┤ Tables ───┤ Charts ───┘
↓ Multimodal ingestion ↓ Representation / Embeddings ↓ Indexes ↓
Multimodal retrieval ↓ LLM / VLM ↓ Grounded response Playground V4 ---
Multimodal Knowledge Assistant.

PHASE 5 --- FRAMEWORKS, AGENTS & INTEROPERABILITY

Tier 24 --- LangChain & LlamaIndex

Priority: ★★★★★ · CODE · INT · ARCH

Framework fundamentals

24.1 Why AI frameworks exist 24.2 Raw Python vs framework abstractions
24.3 Framework benefits 24.4 Framework limitations 24.5 Abstraction
leakage 24.6 Vendor/framework lock-in 24.7 When not to use a framework

LangChain fundamentals

24.8 LangChain architecture 24.9 Chat models 24.10 Provider integrations
24.11 Messages 24.12 Prompt templates 24.13 Output parsers 24.14
Structured outputs 24.15 Runnable abstraction 24.16 RunnableSequence
24.17 RunnableParallel 24.18 Composition/pipelines 24.19 Streaming 24.20
Async execution 24.21 Document loaders 24.22 Text splitters 24.23
Embedding integrations 24.24 Vector-store integrations 24.25 Retrievers
24.26 Tools 24.27 Middleware/hooks 24.28 State/history integration 24.29
Error handling 24.30 Tracing

LlamaIndex

24.31 LlamaIndex architecture 24.32 Documents 24.33 Nodes 24.34 Metadata
24.35 Transformations 24.36 Ingestion pipeline 24.37 Indexes 24.38
Vector indexes 24.39 Retrievers 24.40 Query engines 24.41 Response
synthesizers 24.42 Citation handling 24.43 Routers 24.44 Structured-data
integration 24.45 Evaluation concepts

Framework comparison

24.46 Raw Python vs LangChain 24.47 LangChain vs LlamaIndex 24.48
Framework interoperability 24.49 Dependency management 24.50 Framework
versioning 24.51 Debugging framework abstractions 24.52 Production
considerations

Build

Take the RAG system we already built manually and implement it three
ways: Same requirements │ ┌────┼──────────────┐ ↓ ↓ ↓ Raw LangChain
LlamaIndex Python │ │ │ └─────┼──────────────┘ ↓ Compare

Quality Latency Complexity Code volume Debuggability Cost
Maintainability

This prevents you from becoming dependent on framework syntax.

Tier 25 --- LangGraph & Stateful AI Workflows

Priority: ★★★★★ · CORE · CODE · INT · ARCH

Graph fundamentals

25.1 What is LangGraph? 25.2 Why graph-based orchestration? 25.3 Graph
25.4 Node 25.5 Edge 25.6 State 25.7 State schema 25.8 Reducers 25.9
START 25.10 END

Execution

25.11 Sequential execution 25.12 Conditional routing 25.13 Branching
25.14 Parallel execution 25.15 Fan-out 25.16 Fan-in 25.17 Loops 25.18
Cycles 25.19 Termination conditions

State management

25.20 State updates 25.21 Shared state 25.22 Private state 25.23 Message
state 25.24 Persistence 25.25 Checkpointing 25.26 Durable execution
25.27 Resume/recovery 25.28 Thread/session identity

Human-in-the-loop

25.29 Interrupts 25.30 Human approval 25.31 Human correction 25.32
Resume after approval 25.33 Escalation

Reliability

25.34 Node failure 25.35 Retry policies 25.36 Timeout handling 25.37
Fallback nodes 25.38 Recovery 25.39 Idempotency

Advanced graph design

25.40 Subgraphs 25.41 Nested workflows 25.42 Dynamic routing 25.43
Streaming state 25.44 Tool nodes 25.45 Agent nodes 25.46 RAG nodes 25.47
Supervisor nodes 25.48 Graph visualization 25.49 Graph tracing

Architectural decisions

25.50 Workflow vs agent 25.51 Deterministic vs probabilistic routing
25.52 State machine vs autonomous agent 25.53 When LangGraph is
unnecessary 25.54 Production workflow design

Build

Playground V5 --- Visual Stateful Workflow Runtime

START ↓ Classify ↓ ┌───────────────┐ │ │ ↓ ↓ RAG SQL │ │
└──────┬────────┘ ↓ Review ↓ Confidence? ↙ ↘ Low High ↓ ↓ Human Answer ↓
↓ └──────────→ END

This directly connects with your visual AI-system builder.

⸻

Tier 26 --- AI Agents

Priority: ★★★★★ · CORE

Fundamentals

26.1 What is an AI agent? 26.2 LLM application vs agent 26.3 Workflow vs
agent 26.4 Agent environment 26.5 Goals 26.6 Observations 26.7 Actions
26.8 Tools 26.9 State 26.10 Memory

Agent loop

26.11 Observe 26.12 Reason/decide 26.13 Act 26.14 Observe result 26.15
Continue/terminate

Conceptually:

          ┌─────────────────────┐
          │                     │
          ↓                     │
       Observe                   │
          ↓                     │
      Decide/Plan                │
          ↓                     │
     Select Action               │
          ↓                     │
      Execute Tool               │
          ↓                     │
    Observe Result ──────────────┘
          │
       Complete?
          ↓
        Answer

Agent patterns

26.16 ReAct 26.17 Plan-and-execute 26.18 Tool-using agent 26.19 Routing
agent 26.20 Planning agent 26.21 Reflection 26.22 Critique 26.23
Self-correction 26.24 Verification 26.25 Reviewer pattern

Planning

26.26 Task decomposition 26.27 Subtasks 26.28 Planning horizon 26.29
Dynamic replanning 26.30 Goal tracking 26.31 Plan validation

Reliability

26.32 Agent loops 26.33 Infinite loops 26.34 Step limits 26.35 Token
budgets 26.36 Cost budgets 26.37 Time budgets 26.38 Tool failure 26.39
Recovery 26.40 Termination criteria

Safety

26.41 Agent permissions 26.42 Tool authorization 26.43 Action validation
26.44 Human approval 26.45 Sandboxing 26.46 Auditability

Evaluation

26.47 Task completion 26.48 Tool correctness 26.49 Planning quality
26.50 Number of steps 26.51 Agent latency 26.52 Agent cost 26.53 Failure
rate

Build

Single-Agent Laboratory

User → Agent → Tools → Verification → Response.

⸻

Tier 27 --- Multi-Agent Systems

Priority: ★★★★★ · CODE · ARCH · INT

Fundamentals

27.1 What is a multi-agent system? 27.2 Single vs multi-agent 27.3 Agent
specialization 27.4 Roles 27.5 Responsibilities 27.6 Agent boundaries

Architectures

27.7 Supervisor-worker 27.8 Router architecture 27.9 Hierarchical agents
27.10 Sequential agents 27.11 Parallel agents 27.12 Peer-to-peer agents
27.13 Blackboard/shared workspace 27.14 Debate pattern 27.15
Generator-reviewer 27.16 Planner-executor 27.17
Researcher-writer-reviewer

Communication

27.18 Agent messages 27.19 Handoffs 27.20 Delegation 27.21 Task
assignment 27.22 Shared context 27.23 Isolated context 27.24 Shared
memory 27.25 Private memory 27.26 Shared state

Coordination

27.27 Orchestration 27.28 Choreography 27.29 Scheduling 27.30 Parallel
execution 27.31 Synchronization 27.32 Fan-out/fan-in 27.33 Result
aggregation

Failure modes

27.34 Agent disagreement 27.35 Conflicting actions 27.36 Context
contamination 27.37 Cascading hallucinations 27.38 Infinite delegation
27.39 Deadlocks 27.40 Duplicate work 27.41 Coordination overhead

Control

27.42 Maximum delegation depth 27.43 Step budgets 27.44 Cost budgets
27.45 Timeout budgets 27.46 Supervisor override 27.47 Human escalation

Evaluation

27.48 Team task success 27.49 Agent contribution 27.50 Delegation
accuracy 27.51 Coordination efficiency 27.52 Total latency 27.53 Total
token consumption 27.54 Cost per completed task

Build

Your first serious multi-agent system:

                    Supervisor
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓

Research Agent RAG Agent SQL Agent │ │ │
└───────────────┼───────────────┘ ↓ Reviewer Agent ↓ Supervisor ↓ User

Later we'll add coding agents.

⸻

Tier 28 --- Model Context Protocol (MCP)

Priority: ★★★★★ · CORE · ARCH · CODE · INT

This deserves a full tier rather than being buried inside agents.

Fundamentals

28.1 Why interoperability is needed 28.2 What is MCP? 28.3 MCP
architecture 28.4 MCP host 28.5 MCP client 28.6 MCP server 28.7
Capability negotiation

MCP primitives

28.8 Tools 28.9 Resources 28.10 Prompts 28.11 Tool discovery 28.12
Resource discovery 28.13 Prompt discovery 28.14 Tool invocation 28.15
Resource access

Communication

28.16 Messages 28.17 Requests 28.18 Responses 28.19 Notifications 28.20
Session lifecycle 28.21 Transport concepts 28.22 Local communication
28.23 Remote communication considerations

Server development

28.24 Building an MCP server 28.25 Exposing Python functions 28.26
File-resource server 28.27 Database server 28.28 API server 28.29 Search
server 28.30 Enterprise service wrapper

Security

28.31 Authentication 28.32 Authorization 28.33 Least privilege 28.34
Resource permissions 28.35 Tool permissions 28.36 Secret isolation 28.37
Trust boundaries 28.38 Input validation 28.39 Output validation 28.40
Audit logging

Production

28.41 MCP observability 28.42 MCP error handling 28.43 Timeouts 28.44
Retries 28.45 Version compatibility 28.46 Server lifecycle 28.47 Scaling
considerations

Architecture comparison

28.48 MCP vs function calling 28.49 MCP vs REST API 28.50 MCP vs plugin
architecture 28.51 MCP vs A2A

Build

Create multiple MCP servers:

AI Playground │ MCP Client │ ┌────┼─────────────┬──────────────┐ ↓ ↓ ↓ ↓
Files SQL Search Project MCP MCP MCP MCP

This also fits your visual system builder extremely well.

⸻

Tier 29 --- Agent-to-Agent Interoperability

Priority: ★★★★☆

This tier should remain more architecture-focused because
interoperability standards can evolve quickly.

Foundations

29.1 Why agent interoperability? 29.2 Agent-to-agent communication 29.3
Agent identity 29.4 Agent discovery 29.5 Capability discovery 29.6 Agent
metadata 29.7 Agent cards/capability descriptions

Tasks

29.8 Task creation 29.9 Task delegation 29.10 Task lifecycle 29.11 Task
status 29.12 Long-running tasks 29.13 Asynchronous tasks 29.14
Cancellation 29.15 Result delivery

Communication

29.16 Messages 29.17 Context sharing 29.18 Artifact exchange 29.19
Streaming updates 29.20 Error communication

Enterprise concerns

29.21 Authentication 29.22 Authorization 29.23 Agent trust 29.24
Identity propagation 29.25 Auditability 29.26 Cross-system communication
29.27 Version compatibility 29.28 Failure recovery

Architectural comparison

29.29 A2A vs MCP 29.30 A2A vs REST 29.31 A2A vs message queues 29.32
A2A + MCP together 29.33 Multi-agent framework vs interoperability
protocol

Build

Enterprise Agent A │ │ Agent communication ↓ Enterprise Agent B │ │ MCP
↓ ┌─────┼─────────┐ ↓ ↓ ↓ SQL Files APIs

The conceptual distinction to remember will eventually be:

MCP connects an AI application/agent to capabilities and context;
A2A-style interoperability connects independently operating
agents/systems to one another.

We will be much more precise about the standards and their current
specifications when writing the actual chapter.

⸻

PHASE 6 --- ENTERPRISE AI ENGINEERING

Now we move from:

"Can I build an AI system?"

to:

"Can this AI system safely serve thousands of real users?"

That distinction is extremely important for your target roles.

Tier 30 --- AI System Evaluation

Priority: ★★★★★ · CORE · INT · PROD

Evaluation foundations

30.1 Why AI evaluation exists 30.2 Offline evaluation 30.3 Online
evaluation 30.4 Human evaluation 30.5 Automated evaluation 30.6 Golden
datasets 30.7 Benchmark datasets 30.8 Evaluation datasets 30.9
Regression datasets

Model evaluation

30.10 Accuracy 30.11 Relevance 30.12 Correctness 30.13 Factuality 30.14
Hallucination 30.15 Completeness

Retrieval evaluation

30.16 Recall@K 30.17 Precision@K 30.18 MRR 30.19 NDCG 30.20 Hit rate
30.21 Context relevance

RAG evaluation

30.22 Retrieval quality 30.23 Context precision 30.24 Context recall
30.25 Faithfulness 30.26 Groundedness 30.27 Answer relevance 30.28
Citation correctness

Tool evaluation

30.29 Tool selection 30.30 Argument correctness 30.31 Execution success
30.32 Tool result utilization

Agent evaluation

30.33 Task success 30.34 Planning quality 30.35 Number of steps 30.36
Tool efficiency 30.37 Recovery ability 30.38 Termination correctness

Multi-agent evaluation

30.39 Delegation quality 30.40 Coordination quality 30.41 Individual
contribution 30.42 End-to-end task success

LLM-as-judge

30.43 Judge models 30.44 Evaluation prompts 30.45 Pairwise evaluation
30.46 Scoring evaluation 30.47 Judge bias 30.48 Position bias 30.49
Calibration 30.50 Human verification

Production evaluation

30.51 A/B testing 30.52 Shadow testing 30.53 Canary evaluation 30.54
User feedback 30.55 Regression testing 30.56 Quality gates

Build

AI Evaluation Harness

Every Playground architecture should eventually run through the same
evaluation framework.

⸻

Tier 31 --- AI Observability & LLMOps

Priority: ★★★★★

Observability

31.1 Logs 31.2 Metrics 31.3 Traces 31.4 Spans 31.5 Distributed tracing
31.6 Correlation IDs 31.7 OpenTelemetry concepts 31.8 OpenInference
concepts

LLM metrics

31.9 Prompt tokens 31.10 Completion tokens 31.11 Total tokens 31.12 TTFT
31.13 End-to-end latency 31.14 Tokens/second 31.15 Throughput 31.16
Error rate 31.17 Cost/request

RAG traces

31.18 Query trace 31.19 Embedding trace 31.20 Retrieval trace 31.21
Reranking trace 31.22 Generation trace

Agent traces

31.23 Agent step trace 31.24 Tool-call trace 31.25 Handoff trace 31.26
State transitions 31.27 Failure traces

LLMOps

31.28 Prompt registry 31.29 Prompt versions 31.30 Model versions 31.31
Dataset versions 31.32 Experiment tracking 31.33 Evaluation history
31.34 Deployment versions 31.35 Feedback loops

Platforms/concepts

31.36 Phoenix 31.37 Langfuse 31.38 Arize 31.39 Dynatrace AI
observability concepts 31.40 Cloud-native monitoring SLI, SLO telemetry
relationship, trace sampling, log sampling, PII-safe telemetry,
dashboards, alerting, anomaly detection.

Build

This becomes the live inspector in your visual AI runtime:

REQUEST │ ├── Prompt 12 ms ├── Embedding 37 ms ├── Retrieval 74 ms ├──
Reranker 116 ms ├── LLM 1,420 ms └── Output 8 ms TOTAL 1,667 ms Input
tokens 2,481 Output tokens 416 Tool calls 2 Estimated cost \$...

⸻

Tier 32 --- AI Security, Guardrails & Governance

Priority: ★★★★★ · SEC · PROD · INT

Threats

32.1 AI threat modeling 32.2 Attack surface 32.3 Prompt injection 32.4
Direct prompt injection 32.5 Indirect prompt injection 32.6 Jailbreaking
32.7 Data exfiltration 32.8 Tool abuse 32.9 Privilege escalation 32.10
Poisoned retrieval content 32.11 Memory poisoning 32.12 Agent
manipulation

Data security

32.13 PII 32.14 Sensitive data 32.15 Redaction 32.16 Data minimization
32.17 Encryption 32.18 Secrets 32.19 Tenant isolation

Identity

32.20 Authentication 32.21 Authorization 32.22 RBAC 32.23 ABAC 32.24
Least privilege 32.25 Service identity

RAG security

32.26 Retrieval authorization 32.27 Document permissions 32.28 Metadata
filtering 32.29 Cross-tenant leakage

Agent security

32.30 Tool permissions 32.31 Action validation 32.32 Sandboxing 32.33
Human approval 32.34 Consequential actions 32.35 Agent boundaries

Guardrails

32.36 Input guardrails 32.37 Output guardrails 32.38 Tool guardrails
32.39 Content safety 32.40 Schema validation 32.41 Policy enforcement

Governance

32.42 Audit trails 32.43 Data retention 32.44 Model governance 32.45
Prompt governance 32.46 Dataset governance 32.47 Risk classification
32.48 Approval workflows 32.49 Incident response 32.50 AI system
documentation

Threat-model methodology, OWASP-style AI/LLM risk taxonomy as a
reference rather than something to memorize, supply-chain security,
model provenance, dependency security, egress control, network
isolation, data residency, key rotation.

Build

Secure Enterprise AI Playground

At this point, an answer isn't allowed simply because the LLM wants to
produce it:

User ↓ Identity ↓ Authorization ↓ Input Guardrail ↓ AI System ↓ Tool
Permission ↓ Execution ↓ Output Guardrail ↓ Audit ↓ User

⸻

Tier 33 --- Production AI Backend Engineering

Priority: ★★★★★

Python backend

33.1 Application architecture 33.2 FastAPI 33.3 Pydantic 33.4 Dependency
injection 33.5 Configuration management 33.6 Environment management

APIs

33.7 REST 33.8 HTTP methods 33.9 Status codes 33.10 Request validation
33.11 Response validation 33.12 API versioning 33.13 Pagination 33.14
Idempotency

Async

33.15 Synchronous programming 33.16 Asynchronous programming 33.17
async/await 33.18 Event loop 33.19 I/O-bound workloads 33.20 Concurrency

Streaming

33.21 SSE 33.22 WebSockets 33.23 Streaming responses 33.24 Cancellation
33.25 Backpressure

Reliability

33.26 Timeout 33.27 Retry 33.28 Backoff 33.29 Circuit breaker 33.30
Bulkhead pattern 33.31 Graceful degradation 33.32 Fallback

Infrastructure components

33.33 Redis 33.34 Caching 33.35 Queues 33.36 Workers 33.37 Background
jobs 33.38 Event-driven processing

API protection

33.39 Authentication 33.40 Authorization 33.41 Rate limiting 33.42
Quotas 33.43 API gateway

Build

Turn the Playground into a proper service rather than a collection of
scripts.

⸻

Tier 34 --- Containers, Testing, CI/CD & Infrastructure as Code

Containers

34.1 Container concepts 34.2 Docker 34.3 Dockerfile 34.4 Images 34.5
Containers 34.6 Layers 34.7 Build context 34.8 Multi-stage builds 34.9
Container registries 34.10 Docker Compose

Testing

34.11 Unit testing 34.12 Integration testing 34.13 End-to-end testing
34.14 Contract testing 34.15 AI regression testing 34.16 Load testing
34.17 Security testing

CI

34.18 Git workflow 34.19 Pull requests 34.20 Linting 34.21 Formatting
34.22 Type checking 34.23 Test automation 34.24 Security scanning 34.25
Image scanning

CD

34.26 Build artifacts 34.27 DEV 34.28 QA/Test 34.29 Staging 34.30
Production 34.31 Deployment strategies 34.32 Rolling deployment 34.33
Blue-green 34.34 Canary 34.35 Rollback

Infrastructure as Code

34.36 IaC concepts 34.37 Terraform 34.38 Providers 34.39 Resources 34.40
Variables 34.41 Outputs 34.42 Modules 34.43 State 34.44 Remote state
34.45 Environment separation 34.46 Secret integration

Build

Developer ↓ Feature Branch ↓ Pull Request ↓ Lint / Type Check ↓ Unit
Tests ↓ Integration Tests ↓ RAG / Agent Evaluation ↓ Security Scan ↓
Docker Build ↓ Registry ↓ DEV ↓ Smoke Tests ↓ Canary ↓ PROD

⸻

Tier 35 --- Cloud AI Engineering

Rather than memorize hundreds of cloud product names, this tier will
teach the architectural capability first, then map it to GCP/Azure/AWS.

Core cloud concepts

35.1 Regions/zones 35.2 Projects/accounts/subscriptions 35.3 IAM 35.4
Service identities 35.5 Networking 35.6 Private networking 35.7 Storage
35.8 Databases 35.9 Data warehouses 35.10 Containers 35.11 Serverless
35.12 Kubernetes 35.13 Queues/events 35.14 Secret management 35.15
Monitoring 35.16 Logging 35.17 Managed AI platforms 35.18 Managed model
endpoints 35.19 GPU infrastructure

GCP mapping

Vertex AI, Cloud Run, GKE, BigQuery, Cloud Storage, Pub/Sub, Secret
Manager, IAM, Artifact Registry, Cloud Monitoring.

Azure mapping

Azure AI services, Azure OpenAI concepts, Azure Machine Learning,
Databricks, ADLS, Functions/Container Apps/AKS, Key Vault, Monitor.

AWS mapping

Bedrock, SageMaker, Lambda, ECS/EKS, S3, SQS/SNS/EventBridge, Secrets
Manager, CloudWatch.

Build

Deploy the same logical AI architecture on at least one primary cloud,
then learn how it maps to the other two.

Given your current data-engineering background, GCP can remain the
deepest implementation, while Azure becomes particularly important for
the Databricks/AI job scenarios.

⸻

Tier 36 --- AI Performance, Scalability, Reliability & Cost Engineering

Priority: ★★★★★

This is important enough to remain its own tier.

Performance

36.1 Latency 36.2 TTFT 36.3 Inter-token latency 36.4 Throughput 36.5
Concurrency 36.6 P50 36.7 P95 36.8 P99

Optimization

36.9 Caching 36.10 Prompt caching 36.11 Semantic caching 36.12
Connection pooling 36.13 Parallel execution 36.14 Async execution 36.15
Batching 36.16 Continuous batching 36.17 Quantization 36.18 Model
routing 36.19 Smaller-model routing

Scaling

36.20 Vertical scaling 36.21 Horizontal scaling 36.22 Autoscaling 36.23
Load balancing 36.24 Queue-based scaling 36.25 GPU utilization

Reliability

36.26 Availability 36.27 SLA 36.28 SLO 36.29 Error budget 36.30 High
availability 36.31 Fault tolerance 36.32 Disaster recovery 36.33
Multi-region concepts 36.34 Provider fallback 36.35 Model fallback

Cost

36.36 Cost/request 36.37 Cost/token 36.38 Cost/user 36.39 GPU cost 36.40
Retrieval cost 36.41 Agent cost amplification 36.42 Budget enforcement
36.43 Cost-quality trade-offs 36.44 Cost-latency trade-offs Capacity
planning Load testing methodology Stress testing Soak testing
Concurrency limits Queue depth Rate-limit budgeting GPU memory budgeting
Token throughput budgeting

Build

This is where the Ask Anna Performance Lab becomes especially useful.

We deliberately create:

Slow System ↓ Measure ↓ Find bottleneck ↓ Optimize ↓ Load test ↓ Measure
again ↓ Compare

⸻

PHASE 7 --- ENTERPRISE ARCHITECTURE & JOB PREPARATION

Tier 37 --- AI System Design

Priority: ★★★★★ · ARCH · INT

Requirements

37.1 Business problem 37.2 Stakeholders 37.3 Functional requirements
37.4 Non-functional requirements 37.5 Constraints 37.6 Assumptions

Architecture

37.7 HLD 37.8 LLD 37.9 Component boundaries 37.10 Data flow 37.11 API
contracts 37.12 Sequence diagrams 37.13 State diagrams

AI decisions

37.14 Model selection 37.15 Hosted vs local 37.16 RAG vs fine-tuning
37.17 Workflow vs agent 37.18 Single vs multi-agent 37.19 Vector vs SQL
vs graph 37.20 Build vs buy

Enterprise NFRs

37.21 Latency 37.22 Throughput 37.23 Availability 37.24 Scalability
37.25 Reliability 37.26 Security 37.27 Compliance 37.28 Observability
37.29 Maintainability 37.30 Cost

Delivery

37.31 POC 37.32 Prototype 37.33 Pilot 37.34 MVP 37.35 Production 37.36
Rollout 37.37 Feedback 37.38 Iteration

Architecture documentation

37.39 ADR 37.40 Trade-off documentation 37.41 Runbooks 37.42 Client
explanation 37.43 Executive explanation 37.44 Engineering explanation

Architecture trade-off analysis Failure-mode analysis Capacity estimates
Threat model Cost model Dependency mapping Data-flow diagrams
Trust-boundary diagrams ⸻

Tier 38 --- AI Data Engineering

Priority: ★★★★★ for your target roles

Sources and ingestion

38.1 Structured data 38.2 Semi-structured data 38.3 Unstructured data
38.4 Batch ingestion 38.5 Streaming ingestion 38.6 APIs 38.7 Files 38.8
Databases 38.9 Events

Data architecture

38.10 Data lake 38.11 Warehouse 38.12 Lakehouse 38.13 Bronze 38.14
Silver 38.15 Gold 38.16 Staging 38.17 Curated

Processing

38.18 ETL 38.19 ELT 38.20 SQL 38.21 Python 38.22 Spark 38.23 Distributed
processing 38.24 Incremental processing 38.25 CDC

Data quality

38.26 Completeness 38.27 Accuracy 38.28 Validity 38.29 Consistency 38.30
Uniqueness 38.31 Freshness 38.32 Data contracts

AI-ready data

38.33 Document pipelines 38.34 Embedding pipelines 38.35 Vector-index
pipelines 38.36 Knowledge-graph pipelines 38.37 Evaluation datasets
38.38 Training datasets 38.39 Metadata pipelines 38.40 Feature/semantic
layers

Governance

38.41 Lineage 38.42 Catalog 38.43 Access control 38.44 PII 38.45
Retention 38.46 Audit

Platforms

38.47 BigQuery for AI systems 38.48 Databricks for AI systems 38.49 Data
lake + RAG 38.50 Warehouse + agents

Orchestration fundamentals DAGs Scheduling Dependencies Retries
Backfills Reprocessing Idempotent pipelines Schema evolution
Partitioning Clustering Data skew Small-file problem Late-arriving data
Event time vs processing time Watermarks This tier bridges your Data
Engineering experience with AI Engineering, rather than treating them as
unrelated careers.

⸻

Tier 39 --- Enterprise AI Architecture & Design Patterns

Core patterns

39.1 Model Gateway 39.2 Provider Adapter 39.3 Model Router 39.4 Prompt
Registry 39.5 RAG Service 39.6 Retrieval Gateway 39.7 Semantic Layer
39.8 Tool Gateway 39.9 Agent Orchestrator 39.10 Supervisor-Agent 39.11
Human Approval 39.12 Guardrail Gateway 39.13 Evaluation Gateway 39.14
Observability Layer

Reliability patterns

39.15 Retry 39.16 Circuit breaker 39.17 Fallback 39.18 Bulkhead 39.19
Timeout 39.20 Graceful degradation

AI-specific patterns

39.21 Small-model → large-model escalation 39.22 Deterministic-first
routing 39.23 Retrieval-before-generation 39.24
Verification-before-action 39.25 Human-before-consequence 39.26
Agent-as-tool 39.27 Evaluator-agent 39.28 Planner-executor 39.29
Generator-reviewer 39.30 Event-driven agent

Enterprise integration

39.31 APIs 39.32 Events 39.33 MCP 39.34 Agent interoperability 39.35
Databases 39.36 Enterprise search 39.37 Identity systems 39.38 Existing
applications

⸻

Tier 40 --- Enterprise Capstones & Interview Mastery

This is where everything converges.

Project 1 --- Raw AI Engineering Playground

Providers → prompts → memory → tools → metrics.

Project 2 --- Enterprise RAG Platform

Hybrid retrieval → reranking → evaluation → security → observability.

Project 3 --- Workforce Intelligence AI

Workforce Data + HR Policies ↓ Data Engineering ↓ SQL + RAG ↓ Agent
Router ┌────┼─────┐ ↓ ↓ ↓ SQL RAG Tools └────┼─────┘ ↓ Guardrails ↓
FastAPI ↓ Evaluation

Project 4 --- Ask Anna Performance Lab

Azure + Databricks + Genie + Spark + LLM + RAG + API performance.

Project 5 --- Multimodal Enterprise Assistant

PDF + images + screenshots + tables + audio + video.

Project 6 --- Multi-Agent Operations System

Supervisor + specialists + tools + memory + HITL + evaluation.

Project 7 --- MCP/A2A Enterprise Platform

Agents + standardized capabilities + enterprise services.

Project 8 --- AI Platform Engineering

Applications ↓ AI Gateway ↓ Model Router ┌────┼────────┬────────┐ ↓ ↓ ↓
↓ Open Claude Gemini Local AI Models ↓ RAG / Tools / Agents ↓ Evaluation
↓ Observability ↓ Security

Project 9 --- Secure Enterprise AI Assistant

Identity → permissions → retrieval authorization → tools → HITL → audit.

Project 10 --- Forward-Deployed AI Capstone

Requirements gathering → architecture → prototype → evaluation →
security → deployment → monitoring → optimization → client presentation.

⸻

Final interview layer

Tier 40 also includes:

40.1 Resume explanation 40.2 Project storytelling 40.3 Architecture
explanation 40.4 30-second answers 40.5 60-second answers 40.6 Deep-dive
answers 40.7 Rapid-fire questions 40.8 Coding questions 40.9 Debugging
scenarios 40.10 RAG scenarios 40.11 Agent scenarios 40.12 Data scenarios
40.13 Performance scenarios 40.14 Security scenarios 40.15 Cloud
scenarios 40.16 System-design interviews 40.17 Client conversations
40.18 Trade-off discussions 40.19 Incident/debugging explanations 40.20
Whiteboard architecture

⸻

# CURRICULUM FREEZE --- V1.0

Status: FROZEN

The seven phases and forty tiers are canonical for Version 1.0.

Rules after freeze:

-   Do not add, remove, move or renumber a canonical concept during
    chapter authoring.
-   One concept has one authoritative teaching location; other tiers
    cross-reference it.
-   Durable concepts receive the deepest treatment. Framework-,
    provider-, product- and protocol-specific details are implementation
    examples and must be verified against current documentation before
    publication.
-   If an author or reviewer finds a genuine gap, use
    `PROPOSE_ADDITION:` and do not silently insert it.
-   A proposed curriculum change must document: reason, affected tiers,
    dependency impact, duplication impact, terminology impact, and
    approve/reject decision.
-   Approved curriculum changes increment the roadmap version (for
    example V1.1).
-   P0/P1 factual, security or dependency defects may block publication
    even when the curriculum itself is frozen.
-   Chapter status is tracked separately in `CHAPTER_STATUS.md`;
    freezing the curriculum does not mean chapters are written or final.

The roadmap is now FROZEN as Version 1.0 and ready for chapter authoring

We now have the complete backbone:

PHASE 1 --- Deep Learning Foundations T01 Mathematics T02 Machine
Learning T03 Neural Networks T04 Training & Optimization T05 Sequence
Models PHASE 2 --- Transformers & Foundation Models T06 Tokenization &
Embeddings T07 Attention T08 Transformers T09 Pretraining T10
Fine-Tuning & Alignment T11 Inference T12 Modern Architectures T13
Vision T14 Audio/Video T15 Model Evaluation PHASE 3 --- LLM Application
Engineering T16 Raw LLM Engineering T17 Prompt/Context Engineering T18
Memory T19 Structured Output & Tools PHASE 4 --- RAG & Multimodal T20
RAG T21 Advanced RAG T22 GraphRAG/Agentic RAG T23 Multimodal PHASE 5 ---
Agents & Interoperability T24 LangChain/LlamaIndex T25 LangGraph T26
Agents T27 Multi-Agent T28 MCP T29 A2A PHASE 6 --- Enterprise AI T30
Evaluation T31 Observability/LLMOps T32 Security T33 Backend T34
CI/CD/IaC T35 Cloud T36 Performance/Scale/Cost PHASE 7 --- Enterprise
Architecture T37 System Design T38 AI Data Engineering T39 Design
Patterns T40 Capstones + Interview Mastery

This gives us roughly 1,000+ individual concepts/subconcepts once every
compressed tier is fully expanded. That reinforces why the final result
is closer to a serious technical handbook than a traditional cheat
sheet.

The next step should no longer be more roadmap planning. We should start
writing the actual book at Tier 1. For Tier 1, I'll apply the agreed
23-part format, keep small concepts concise and major concepts deep,
include tested Python where useful, equations, examples, enterprise/AI
connections, interview explanations, rapid-fire answers, exercises, and
memory hooks. After Tier 1 is complete, we produce its condensed
revision sheet and only then advance to Tier 2.

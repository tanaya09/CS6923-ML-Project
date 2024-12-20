# Comparative Analysis of BERT and GPT Architectures

This project implements and analyzes BERT and GPT language models through both training from scratch and fine-tuning approaches, providing insights into their architectural differences, strengths, and performance characteristics.

## Project Overview

- Implemented BERT with scaled-down architecture (8 layers, 256 hidden dimensions)
- Implemented GPT-2 (12 layers, 768 hidden dimensions) 
- Training on WikiText-103 dataset
- Fine-tuning experiments using SQuAD and CNN/DailyMail datasets
- Comprehensive comparison of architecture performance and capabilities

## Implementation Details

### Base Models
- **BERT**: 
  - 8 transformer layers
  - 256 hidden dimensions
  - Masked Language Modeling & Next Sentence Prediction
  - Training loss: Close to random chance on NSP tasks

- **GPT**: 
  - 12 transformer layers
  - 768 hidden dimensions
  - Autoregressive language modeling
  - Validation loss: 3.1 after 15 epochs

### Fine-tuned Models
- **Fine-tuned BERT**:
  - Based on DistilBERT
  - 6 transformer layers
  - Evaluation loss: 1.67
  - Processing speed: 403.26 samples/second

- **Fine-tuned GPT**:
  - Based on GPT-2
  - Training loss: 2.8
  - Evaluation loss: 2.7
  - Perplexity: 15.4

## Datasets

1. **WikiText-103**:
   - Base training dataset
   - 41 million tokens (40% of original)
   - Context window: 1024 tokens

2. **SQuAD**:
   - Question-answering dataset
   - 5000 examples (80-20 split)
   - Used for BERT fine-tuning

3. **CNN/DailyMail**:
   - Summarization dataset
   - Used for GPT fine-tuning
   - Combined with SQuAD for multi-task training

## Key Results

### BERT Performance
- Base model showed limitations in from-scratch training
- Fine-tuned model achieved strong QA performance
- Effective at precise information extraction
- Superior at analytical tasks

### GPT Performance
- Strong generative capabilities
- Effective at both summarization and QA tasks
- Generated coherent and contextually relevant text
- Could generate follow-up questions

## Requirements

```
torch>=1.7.0
transformers>=4.5.0
datasets
wandb
numpy
pandas
evaluate
```

## Key Findings

1. Training from scratch requires significant computational resources
2. Fine-tuning provides efficient path for specific applications
3. BERT excels at analytical tasks and precise information extraction
4. GPT shows strength in generation and creative tasks
5. Architecture choice should be guided by specific use case requirements

## Future Directions

- Explore hybrid approaches combining strengths of both architectures
- Investigate more efficient training methods
- Expand to larger datasets
- Optimize for specific domain applications

## Contributors
- Aakash Shankar (as17872)
- Subhash Kotaru (sk12154)
- Tanaya Pawar (tp2623)

## References
1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. Language Models are Unsupervised Multitask Learners
3. Other implementation references as listed in the paper


*Note: This is a research project implemented as part of CSGY 6923 Machine Learning course at NYU.*

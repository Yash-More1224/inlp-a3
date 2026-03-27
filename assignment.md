# Assignment Instructions

## General Instructions

1. This assignment must be implemented in Python using Pytorch. You are required to implement the models from scratch (so no nn.RNN or nn.LSTM).
2. Your grade will depend on the correctness of your implementation, the quality of your code, the validity of your output, your interpretation of the output and the methods used, and the clarity of your explanations.
3. You are required to push the assignment to GitHub Classroom, there will be no Moodle link for this assignment. You are also required to maintain WandB logs and push your models to HuggingFace. These links must be added to your report.
4. Make sure to test your code thoroughly before submission to avoid run-time errors of unexpected behaviour.

---

## Introduction

For decades, two galactic powers have been locked in a silent but devastating war of intelligence: Bisleri’s Collective, a decentralized alliance of planetary systems, and the expansionist Judhanva’s Dominion.

While their fleets clash in asteroid fields and orbital corridors, the true battle is fought in encrypted transmissions, navigation coordinates, fleet movements, supply routes, and strategic commands transmitted across light-years.

Recently, the Collective intercepted a high-priority Dominion communication stream. The signal was captured and stored as:

* cipher_00.txt: an encrypted version of a classified Dominion directive
* plain.txt: archived linguistic data recovered from previous Dominion leaks

However, the Dominion employs a cipher protocol unknown to the Collective. The intercepted transmission must be decrypted to reveal its contents. As a cryptographic analyst for Bisleri’s Collective, you are tasked with designing neural sequence models, specifically a Recurrent Neural Network (RNN) and a Long Short-Term Memory network (LSTM), to learn the mapping from encrypted text to original plaintext. Beyond decryption, the Collective needs to understand how the Dominion structures its language, so you will also build a State Space Model (SSM) and a Bidirectional LSTM (Bi-LSTM) to model their linguistic patterns. Finally, you must employ a combination of these models to rectify errors and reconstruct the original message in case of solar storms and ion interference corrupting the transmissions.

---

## Implementation Tasks

### 0. Instructions

1. Ensure that your code is well-structured, clean and modular. Do not dump your code into a singular file.
2. Remember to log your training and validation losses, provide samples to show the working, and push the models to Hugging Face.
3. Note that we will be testing the generalisation of your model on unseen data, so ensure that you choose the data splits and hyper-parameters wisely to prevent over-fitting.
4. Include the deliverables mentioned in each part in your final report, and ensure that you provide a detailed analysis of the results obtained.
5. Before starting with the tasks, ensure that you pre-process the data as required.

---

## 1. Decrypting the Cipher

The Collective’s first objective is clear: break the Dominion’s cipher. The fate of an entire star system may depend on whether your model can accurately reconstruct the hidden message.

You are provided with two files:

* plain.txt: Contains English text in its original form.
* cipher_00.txt: Contains the same text encrypted using a simple cipher.

### Objective

Implement and train two sequence models:

* Recurrent Neural Network (RNN)
* Long Short-Term Memory network (LSTM)

Train both models to learn the mapping from cipher text to plain text and recover the original message.

### Metrics to be Used

* Character-level accuracy
* Word-level accuracy
* Leveshtein Distance

### Deliverables

* Quantitative evaluation of decryption performance
* Comparison between RNN and LSTM
* Brief analytical discussion of results

---

## 2. Next Word Prediction and Masked Language Modeling

With the cipher broken, the Collective turns to a deeper objective: understanding how the Dominion structures its language to anticipate future attacks.

Using the same plain.txt corpus, implement and compare two language modeling approaches:

* State Space Model (SSM) for Next Word Prediction (NWP)
* Bidirectional LSTM (Bi-LSTM) for Masked Language Modeling (MLM)

### Metrics to be Used

* Perplexity

### Objective

1. Train both models on the corpus.
2. Evaluate both models on their respective task.
3. Compare their predictive performance.

### Deliverables

* Quantitative comparison on prediction metrics
* Discussion of strengths and limitations

---

## 3. Using Language Modeling to Rectify Decryption Errors

The solar storms and ion interference have taken their toll. The intercepted transmissions are riddled with spurious noise and artifacts. Decryption alone can no longer recover the original messages.

For this part, use the noisy encrypted files cipher_0{x}.txt, where x denotes the level of noise added to the cipher text.

### Objective

1. Select one decryption model from Part 1 (RNN or LSTM).
2. Combine it with each model from Part 2 (SSM and Bi-LSTM).
3. Use the language model to correct decryption errors.
4. Conduct two experiments (one per language model).
5. Compare the following three approaches:

   * Decryption model alone
   * Decryption + SSM
   * Decryption + Bi-LSTM

**Note:** You must use the models trained in the previous parts by loading them from your HuggingFace/local checkpoints (both methods should be supported). While you are allowed to fine-tune the models, you are not allowed to train them from scratch again.

### Metrics to be Used

* Metrics from Part 1 and 2
* ROUGE and BLEU scores

You are allowed to use other metrics you deem necessary for this task.

### Deliverables

* Description of experimental setup
* Performance comparison across different noise levels
* Error analysis
* Discussion of how language modeling improves decryption quality

---

## Submission Instructions

Ensure that your codebase follows the following directory structure:

```
Assn_3/
├── src/ # Source code
├── outputs/ # Generated outputs (logs, plots, results, etc.)
│ ├── logs/ # Logging done while training
│ ├── plots/ # Plots for analysis and report
│ └── results/ # Text files containing the outputs of the models
│ ├── task1_rnn.txt # format: {task_name}_{model_name}.txt
│ ├── task1_lstm.txt # Example of file naming convention, DO NOT CHANGE
│ └── ...
├── README.md # Instructions to run the code and any assumptions made
├── Report.pdf # Final report
└── ... # Other files
```

Your report must include a brief overview of the methodology implemented, the configuration used, the results obtained, and your interpretation of the same. Ensure that your analysis is concise and to the point, and that you have provided sufficient evidence to support your claims. You may include tables, plots, or any other visual aids to help defend your claims.

The report must be in PDF format and should be named `<rollnumber>_a3.pdf`. It should not exceed 6 pages (excluding references and the link to your HuggingFace models and WandB logs) and should follow the following format:

* Font size: 12pt
* Font: Times New Roman (or other equivalent sized font. Do not use a smaller font to fit more content, larger are fine if you wish)
* Margins: 2 inch on all sides.

Any deviations from the directory structure (discouraged) must also be adequately justified in the report. Instructions to run your code must be included in the README and must be correct. Note that verbosity shall be penalised, so refrain from unnecessary slop. Do not submit Jupyter notebooks, they will not be graded.

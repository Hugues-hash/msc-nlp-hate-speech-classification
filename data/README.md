# Data

This project uses the **Hate Speech and Offensive Language** dataset from Davidson, Warmsley,
Macy and Weber (2017). It is not committed here.

## What you need

Place the labelled CSV here:

```
data/raw/labeled_data.csv
```

Each row is a tweet with the annotator vote counts (`count`, `hate_speech`, `offensive_language`,
`neither`), the majority `class` label (0 = hate speech, 1 = offensive language, 2 = neither), and
the `tweet` text.

## Where to get it

- Original repository: https://github.com/t-davidson/hate-speech-and-offensive-language
- Also mirrored on Kaggle.

Cite: Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017) "Automated Hate Speech Detection
and the Problem of Offensive Language", *Proceedings of the 11th International AAAI Conference on
Web and Social Media (ICWSM)*.

## Trained model

The fine-tuned LSTM weights (`best_lstm.pt`) and the RoBERTa checkpoint are produced by running
the notebook and are not committed. Re-running the notebook regenerates them.

## Note

The dataset contains tweets with offensive and hateful language, reproduced for research on
detecting such content. It is used here only for that purpose.

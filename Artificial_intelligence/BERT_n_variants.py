#!/usr/bin/env python
# coding: utf-8

# In[3]:


# BERT and its variants - using pre-trained Transformers from Hugging Face
# Dataset: SMS Spam Collection - ../Datasets/spam.csv
#
# Needs: pip install transformers torch
#        (PyTorch is a big download. Models are downloaded automatically from the
#         Hugging Face Hub the first time they are used, so you need internet, and
#         each model is a few hundred MB. They are cached after the first download.)
#
# WHAT IS BERT?
# BERT = Bidirectional Encoder Representations from Transformers.
# It is the encoder half of a Transformer (script 49), trained on a huge amount of
# plain text (Wikipedia + books) with two tasks that need no human labels:
#   1. Masked language modelling: hide ~15% of the words, predict them from context
#   2. Next sentence prediction : are these two sentences consecutive in the text?
# "Bidirectional" = to guess a hidden word it uses the words on BOTH sides.
#
# After that pre-training, you FINE-TUNE it: add a small classification layer on top and
# train briefly on your own labelled data. Because BERT already understands language,
# a few thousand examples are enough to beat models trained from scratch (scripts 47-49).

import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
)

torch.manual_seed(42)
np.random.seed(42)


# In[2]:


get_ipython().run_line_magic('pip', 'install transformers')


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ============================================================
# PART 1 - The tokenizer: how BERT reads text
# ============================================================
# BERT does not split on whole words. It uses "WordPiece": common words stay whole,
# rare words are broken into pieces. This way it never meets a truly unknown word.

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "Unaffordable phones are annoying."
encoded = tokenizer(sentence)

print("\nTokens   :", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
print("Token IDs:", encoded["input_ids"])

# [CLS] is added at the start - its final vector is used to classify the whole sentence
# [SEP] marks the end of a sentence
# pieces starting with ## continue the previous piece (e.g. "un", "##af", "##ford", "##able")

# ---- Batch of sentences: padding and attention mask ----
batch = tokenizer(
    ["Short one.", "This sentence is a little bit longer than the first one."],
    padding=True, return_tensors="pt"
)
print("\ninput_ids shape :", batch["input_ids"].shape)
print("attention_mask  :\n", batch["attention_mask"])
# attention_mask: 1 = real token, 0 = padding. It tells the model to ignore the padding,
# the same masking idea as in script 49.


# In[5]:


# ============================================================
# PART 2 - Masked language modelling: what BERT learned during pre-training
# ============================================================
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

for prompt in ["The capital of France is [MASK].",
               "She poured the [MASK] into a cup of tea."]:
    print(f"\n{prompt}")
    for guess in fill_mask(prompt, top_k=3):
        print(f"   {guess['token_str']:12s} probability {guess['score']:.3f}")

# nobody labelled any of this data, yet BERT picked up facts and grammar just from
# predicting hidden words. That is "self-supervised" learning.


# In[6]:


# PART 3 - Contextual embeddings: the same word gets different vectors
# ============================================================
# Older word embeddings (Word2Vec, GloVe) give "bank" ONE fixed vector, whether it
# means a riverbank or a money bank. BERT's vector for a word depends on its sentence.

bert = AutoModel.from_pretrained("bert-base-uncased")
bert.eval()

sentences = [
    "I deposited my salary at the bank.",         # money bank
    "The bank approved my home loan.",            # money bank
    "We had a picnic on the bank of the river.",  # river bank
]

bank_vectors = []
for s in sentences:
    enc = tokenizer(s, return_tensors="pt")
    with torch.no_grad():
        out = bert(**enc)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    position = tokens.index("bank")
    bank_vectors.append(out.last_hidden_state[0, position])    # BERT's vector for "bank" here

cos = torch.nn.functional.cosine_similarity
print("\nSimilarity of 'bank' vectors (1 = identical meaning):")
print("money bank  vs  money bank :", round(cos(bank_vectors[0], bank_vectors[1], dim=0).item(), 3))
print("money bank  vs  river bank :", round(cos(bank_vectors[0], bank_vectors[2], dim=0).item(), 3))
# you should see the two money-bank vectors closer to each other than to the river bank


# In[7]:


# ============================================================
# PART 4 - Use a ready-made fine-tuned model (no training at all)
# ============================================================
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("\n", sentiment(["I loved this film, it was wonderful!",
                       "The service was slow and the food was cold."]))


# In[11]:


# ============================================================
# PART 5 - Fine-tune DistilBERT on SMS spam
# ============================================================
df = pd.read_csv(r"C:\My-Data-Science-Portfolio\Datasets\spam.csv", encoding="latin-1")
#df = pd.read_csv("../Datasets/spam.csv", encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "message"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"].values, df["label_num"].values,
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

# To keep CPU training to a few minutes, train on a stratified subset.
# On a GPU, set SUBSET = None to use all the training data.
SUBSET = 2000
if SUBSET is not None and SUBSET < len(X_train_text):
    X_train_text, _, y_train, _ = train_test_split(
        X_train_text, y_train, train_size=SUBSET, stratify=y_train, random_state=42
    )
print("\nTraining messages:", len(X_train_text), " Test messages:", len(X_test_text))

MODEL_NAME = "distilbert-base-uncased"     # a smaller, faster BERT (see Part 6)
MAX_LEN = 64                               # SMS messages are short
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5                       # tiny on purpose: we only NUDGE a pre-trained model

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
train_enc = tok(list(X_train_text), truncation=True, padding=True,
                max_length=MAX_LEN, return_tensors="pt")
test_enc = tok(list(X_test_text), truncation=True, padding=True,
               max_length=MAX_LEN, return_tensors="pt")

train_data = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"],
                           torch.tensor(y_train))
test_data = TensorDataset(test_enc["input_ids"], test_enc["attention_mask"],
                          torch.tensor(y_test))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# load pre-trained DistilBERT and attach a fresh 2-class classification layer on top
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1}
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---- Training loop ----
start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # passing labels makes the model return the loss as well
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()           # backpropagation (script 31), through all the layers
        optimizer.step()

        running_loss += loss.item()
        if step % 25 == 0:
            print(f"epoch {epoch + 1}  step {step}/{len(train_loader)}  "
                  f"avg loss {running_loss / step:.4f}")

print(f"Training took {time.time() - start:.0f} seconds")

# ---- Evaluate ----
model.eval()
all_preds = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        logits = model(input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device)).logits
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

print(classification_report(y_test, all_preds, target_names=["ham", "spam"]))

# ---- Save and reuse ----
model.save_pretrained("spam_distilbert")
tok.save_pretrained("spam_distilbert")

spam_classifier = pipeline(
    "text-classification", model="spam_distilbert", tokenizer="spam_distilbert",
    device=device
)
print(spam_classifier([
    "Congratulations! You have won a free prize, call now to claim",
    "Are we still meeting for lunch tomorrow?"
]))
# the folder "spam_distilbert" is a complete model. Load it in the Flask / FastAPI apps
# from scripts 37 and 38 to serve predictions, exactly like the saved .joblib model there.



# In[12]:


# ============================================================
# PART 6 - The BERT family: what each variant changes
# ============================================================
# BERT (2018)    : the original. base = 12 layers, large = 24 layers.
# DistilBERT     : 6 layers instead of 12, made by "knowledge distillation" - a small student
#                  model learns to copy the big BERT. Smaller and faster, keeps most accuracy.
# RoBERTa        : same architecture as BERT, but trained better: more data, longer, bigger
#                  batches, dynamic masking, and it drops next-sentence prediction.
# ALBERT         : shares weights across layers and factorises the embedding matrix, so it has
#                  far fewer parameters (a smaller file, though not proportionally faster).
# ELECTRA        : instead of predicting masked words, it learns to spot which words were
#                  REPLACED by a small generator model. Learns more from the same data.
# DeBERTa        : separates a word's content from its position inside attention; often
#                  among the strongest at this size.
# TinyBERT / MobileBERT : shrunk further for phones and edge devices.
#
# Specialised versions, same idea trained on different text:
#   BioBERT, ClinicalBERT (medical), SciBERT (science papers), FinBERT (finance)
#   multilingual: mBERT, XLM-RoBERTa;  Indian languages: MuRIL, IndicBERT
# (search the Hugging Face Hub for the exact, current model names)

# ---- Compare the variants on real numbers ----
# note: this downloads every model listed, roughly 1.3 GB in total. Remove names to skip.
variants = [
    "distilbert-base-uncased",
    "albert-base-v2",
    "bert-base-uncased",
    "roberta-base",
    ]

test_text = "Unaffordable phones are annoying."
rows = []

for name in variants:
    variant_tok = AutoTokenizer.from_pretrained(name)
    variant_model = AutoModel.from_pretrained(name)
    variant_model.eval()

    n_params = sum(p.numel() for p in variant_model.parameters())

    sample = variant_tok([test_text] * 16, return_tensors="pt", padding=True)
    t0 = time.time()
    with torch.no_grad():
        variant_model(**sample)
    seconds = time.time() - t0

    rows.append({
        "Model": name,
        "Parameters (millions)": round(n_params / 1e6, 1),
        "Time for 16 sentences (s)": round(seconds, 3),
        "Tokens": variant_tok.tokenize(test_text),
    })

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))

# things to notice:
#   - ALBERT has far fewer parameters than BERT, yet its speed is similar, because it still
#     runs the same number of layers - it re-uses weights rather than skipping computation.
#   - DistilBERT is the one that is clearly faster, because it really has half the layers.
#   - Each model splits the same sentence differently: each has its own tokenizer, and a
#     model must always be used with the tokenizer it was trained with.


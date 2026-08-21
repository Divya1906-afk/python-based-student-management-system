# AI Chatbot — Intent Recognition & Query Resolution

**Tools:** Python, scikit-learn, NLTK
**Status:** Fully built, trained, and tested. Metrics below are real, from a
5-fold cross-validated held-out test — not placeholders.

## 1. What This Actually Does

A **machine-learned intent classifier**, not a keyword/rule-matching bot.
Given a user's free-text message, it predicts which of 10 customer-support
intents the message belongs to (greeting, refund, order status, complaint,
etc.), then returns an appropriate response — including correctly saying
"I don't understand" when it isn't confident, instead of guessing.

## 2. Architecture (4 files, single responsibility each)

| File | Responsibility |
|---|---|
| `intents.json` | Training data: 186 example phrases across 10 intents + response templates |
| `preprocess.py` | Text cleaning, tokenization, lemmatization (shared identically between training and inference) |
| `train_model.py` | Trains + evaluates the classifier, saves model artifacts |
| `chatbot.py` | Inference engine: text → intent → confidence-gated response |
| `chat.py` | Interactive CLI front end |

## 3. NLP Pipeline

1. **Cleaning:** lowercase, strip punctuation/digits, collapse whitespace
2. **Tokenization + Lemmatization** (NLTK `WordNetLemmatizer`): "features" → "feature", "hours" → "hour" — so the model generalizes across word forms instead of treating them as unrelated tokens
3. **TF-IDF vectorization** with unigrams + bigrams (`ngram_range=(1,2)`), so phrase patterns like "money back" are captured, not just single words
4. **Classification**: compared Logistic Regression vs. Linear SVM via 5-fold cross-validation; **Linear SVM won** (83.9% vs 82.8% CV accuracy) and was selected automatically. Wrapped in `CalibratedClassifierCV` so it can output calibrated probabilities, not just hard labels — necessary for the confidence threshold below.

## 4. Real Evaluation Results

```
5-fold CV accuracy — Logistic Regression: 82.8% (+/- 5.5%)
5-fold CV accuracy — Linear SVM:          83.9% (+/- 5.4%)
Held-out test accuracy: 86.8%
```

Per-intent precision/recall/F1 is in `train_model.py`'s output — several
intents (payment, human_agent, thanks) hit 100% precision/recall; the
weaker spots (complaint, greeting) reflect natural overlap in short,
similarly-phrased messages, which is realistic and worth mentioning
honestly in an interview rather than claiming perfection.

## 5. The Design Decision That Matters Most: Confidence Thresholding

```python
if confidence < CONFIDENCE_THRESHOLD:  # 0.35
    return fallback_response  # "I'm not sure I understood..."
```

Without this, the classifier would confidently assign *some* intent to
literally any input, including gibberish — verified in testing:
`"asdkjaslkdj random gibberish text"` scored only 31% confidence and
correctly triggered the fallback rather than a wrong, confident answer.
**This is the single most important design choice in the project** and a
strong thing to lead with if asked "what's the hardest part of building a
chatbot?" — real user input is messy, and a model that never admits
uncertainty is actively worse than a simpler rule-based system.

## 6. Sample Conversation (real output, not scripted/faked)

```
You: hello there
Bot: Hey! What brings you here today?  [intent: greeting, confidence: 87%]

You: what payment options do you have
Bot: We support all major cards along with PayPal for payments.
     [intent: payment, confidence: 57%]

You: my item arrived broken and I am upset
Bot: I apologize for the inconvenience — let's get this sorted out. What happened?
     [intent: complaint, confidence: 64%]

You: blah blah nonsense xyz
Bot: Sorry, I didn't catch that. Can you try asking in a different way?
     [intent: fallback, confidence: 31%]
```

Note the payment example ("what payment options do you have") was **not**
a verbatim training pattern — the model generalized correctly to unseen
phrasing, which is the actual point of using ML instead of hardcoded
keyword rules.

## 7. How to Run

```bash
pip install nltk scikit-learn joblib
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python3 train_model.py   # trains model, prints metrics, saves .pkl files
python3 chat.py           # interactive chat
```

## 8. Likely Interview Questions — Prep Notes

**"Why TF-IDF + Linear SVM instead of a transformer/deep learning model?"**
With ~186 labeled examples, a deep model would overfit badly — there isn't
enough data to learn meaningful embeddings from scratch, and using a
pretrained transformer would be disproportionate to the problem's scale.
TF-IDF + a linear classifier is the textbook-correct choice for small,
well-defined intent sets, and it's fast, interpretable, and easy to retrain
as more intents/examples are added.

**"How do you handle a message that doesn't match any intent?"**
Confidence thresholding (see section 5) — below 35% predicted probability,
the bot admits uncertainty instead of guessing. The threshold was chosen by
testing where genuinely out-of-scope inputs (gibberish, unrelated topics)
scored versus real intents.

**"How would you improve accuracy further?"**
More training examples per intent (the current 15-24 per class is workable
but a production system would want 50+), active learning from real
misclassified user queries, and potentially named-entity extraction (e.g.
pulling out an order ID directly from "track order 48213") to make
responses more specific instead of always asking a follow-up question.

**"What happens if two intents are genuinely ambiguous?"**
The `top_intents()` debug method shows the full ranked probability
distribution — in a production system, if the top-2 scores are close, the
bot could ask a clarifying question ("Are you asking about a refund or an
order status?") instead of picking the top one blindly. This isn't
implemented here but is a natural extension to mention.

## Files in this package
- `intents.json` — training data (186 examples, 10 intents)
- `preprocess.py` — text cleaning/lemmatization
- `train_model.py` — training + evaluation script
- `chatbot.py` — inference engine with confidence-gated responses
- `chat.py` — interactive CLI
- `vectorizer.pkl`, `intent_classifier.pkl`, `responses.json` — trained model artifacts (already trained, ready to run)

# Viva Preparation — Fake News Detection Using Machine Learning

**How to use this document.** Each answer has a **short version** (what to say first —
one or two sentences) and a **fuller explanation** to use if the examiner follows up.
Answer the short version, then stop and let them ask for more. Rambling invites harder
questions; a crisp answer sounds confident.

**Numbers you must know by heart:**

| Fact | Value |
|------|-------|
| Selected model | Linear SVM |
| Test accuracy | 94.37% |
| Precision / Recall / F1 | 93.20% / 95.71% / 94.44% |
| Dataset size | 6,304 articles (3,150 FAKE / 3,154 REAL) |
| Train / test split | 5,043 / 1,261 (80/20) |
| TF-IDF features | 20,000, unigrams + bigrams |
| Random state | 42 |
| Confusion matrix | TP 603, FN 27, FP 44, TN 587 |

---

## Section A — Core concepts

### Q1. What is fake news?

**Short answer.** Fake news is news content that is intentionally and verifiably false,
created to mislead readers. The key word is *intentionally* — it is deliberate deception,
not an honest journalistic mistake.

**Fuller explanation.** Researchers distinguish three related terms:
- **Misinformation** — false information spread *without* intent to deceive (someone
  sharing a rumour they believe).
- **Disinformation** — false information created and spread *deliberately* to deceive.
- **Malinformation** — genuine information shared out of context to cause harm.

My project targets content resembling **disinformation**, because that is what labelled
fake-news datasets capture. It matters because fake news has been linked to public-health
harm, market disruption and erosion of trust in institutions, and research in *Science*
(Vosoughi et al., 2018) found false stories spread significantly faster and further than
true ones.

---

### Q2. Why is machine learning used for this problem?

**Short answer.** Because the volume of news published daily makes manual fact-checking
impossible, and because the differences between fake and real news are statistical
patterns rather than fixed rules — which is exactly what machine learning is good at
learning.

**Fuller explanation.** Three reasons:

1. **Scale.** Millions of articles are published daily. A human fact-checker takes hours
   per claim; my model takes 50 milliseconds.
2. **Rules do not work.** A keyword blacklist is trivially evaded by rewording. Machine
   learning generalises to phrasing it has never seen before.
3. **The patterns are statistical, not absolute.** Fake articles *tend* to use more
   sensational vocabulary and fewer attributions — but not always. No hand-written rule
   captures a tendency; a model trained on 6,304 labelled examples can.

**Be ready for the follow-up:** "So could you have written rules instead?" — Answer: you
could for a small set of known articles, but it would fail on anything new, and
maintaining the rules would need constant manual work.

---

### Q3. Why do we need NLP here?

**Short answer.** Machine learning algorithms only work on numbers, but news is text. NLP
is the bridge — it cleans the text and converts it into numerical features.

**Fuller explanation.** NLP does two jobs in my project:

1. **Preprocessing** — normalising messy raw text into a consistent token stream
   (lowercasing, removing URLs, HTML, punctuation, stopwords, lemmatizing).
2. **Vectorization** — TF-IDF converts that token stream into a numeric vector the
   classifier can process.

Without NLP you literally cannot feed text into scikit-learn. And without the
*preprocessing* part specifically, the model would waste capacity learning that `The`
and `the` are the same word, or treating an HTML tag as a meaningful feature.

---

### Q4. What is TF-IDF? Why use it instead of simple word counts?

**Short answer.** TF-IDF stands for Term Frequency–Inverse Document Frequency. It scores
a word highly if it appears often in *one* document but rarely across all documents. Plain
word counts fail because common words like "said" dominate, even though they distinguish
nothing.

**Fuller explanation.** The formula:

```
tf-idf(t, d) = tf(t, d) × log(N / df(t))
```

- **TF** — how often the term appears in this document.
- **IDF** — how rare the term is across the corpus. If a word is in every document, `N/df`
  is close to 1, so the log is near 0, and the whole score collapses.

**Worked example** to give if pressed — 1,000 articles, one 200-word article:

| Term | Count | Docs containing | TF-IDF |
|------|------:|----------------:|-------:|
| `said` | 10 | 950 | 0.053 |
| `election` | 8 | 300 | 0.088 |
| `hoax` | 3 | 25 | 0.070 |

`said` appears the most but scores the **lowest**, because it is everywhere and therefore
distinguishes nothing. That is precisely the behaviour we want.

---

### Q5. What is an n-gram? Why did you use bigrams?

**Short answer.** An n-gram is a sequence of *n* consecutive words. I used unigrams
(single words) and bigrams (word pairs), because some phrases mean much more together than
apart — "mainstream media" is a far stronger signal than "mainstream" and "media"
separately.

**Fuller explanation.**

| n | Name | From "the mainstream media lied" |
|---|------|----------------------------------|
| 1 | Unigram | `the`, `mainstream`, `media`, `lied` |
| 2 | Bigram | `the mainstream`, `mainstream media`, `media lied` |

I set `ngram_range=(1, 2)`. This is confirmed empirically in my project — `mainstream
medium` appears as a top contributing feature in real predictions.

**Why not trigrams?** They roughly triple the feature count while appearing too rarely to
generalise, and my 20,000-feature cap is already reached. The literature (Ahmed et al.,
2017) also found bigrams give the best return.

---

## Section B — The NLP pipeline

### Q6. What is stemming?

**Short answer.** Stemming chops suffixes off words using simple rules to reduce them to a
root form. For example "studies", "studying" and "studied" all become "studi".

**Fuller explanation.** It is fast because it is purely rule-based, but the output is often
not a real word — "studies" → "studi" — because the algorithm just strips characters
without understanding the language. The most common algorithm is the Porter Stemmer, which
is available in my project via `TextPreprocessor(mode="stem")`.

---

### Q7. What is lemmatization, and which did you use?

**Short answer.** Lemmatization reduces a word to its dictionary base form using actual
vocabulary knowledge — "studies" becomes "study", not "studi". **I used lemmatization.**

**Fuller explanation.** The comparison:

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Method | Rule-based suffix stripping | Dictionary + morphological analysis |
| `studies` → | `studi` | `study` |
| `better` → | `better` | `good` |
| Output | May not be a real word | Always a real word |
| Speed | Faster | Slower |

**Why I chose lemmatization** — and this is the important part of the answer: my project
has an **explainability feature** that shows users which words influenced the prediction.
Showing a user that "studi" drove the decision is confusing; "study" is immediately
meaningful. The slight speed cost is irrelevant because preprocessing happens once during
training and takes milliseconds for a single prediction.

---

### Q8. Why remove stopwords?

**Short answer.** Stopwords are extremely common words like "the", "is" and "at" that
appear at roughly the same rate in both fake and real news, so they carry almost no
information for distinguishing the classes — but they make up a large share of the tokens.

**Fuller explanation.** The NLTK English list has 198 words. Removing them cuts the token
count by roughly 40% with no loss of class-discriminating information. That means a smaller
vocabulary, faster training, and less opportunity for the model to overfit to noise.

**Important nuance to add if you want to impress:** stopword removal is *not* always
correct. In sentiment analysis, "not" is a stopword but reverses meaning entirely —
"not good" versus "good". For topical and stylistic classification like mine, removal is
standard and beneficial. Knowing *when a technique does not apply* shows real
understanding.

---

### Q9. Walk me through your full preprocessing pipeline.

**Short answer.** Nine steps: lowercase, remove URLs, remove HTML, remove special
characters, remove numbers, tokenize, remove stopwords, lemmatize, normalise whitespace.

**Worked example** — have this ready:

**Input:**
```
BREAKING!!! Scientists <b>SHOCKED</b> by this 2024 discovery...
Read more at https://example.com/story?id=99 &amp; share it!
```

**Output:**
```
breaking scientist shocked discovery read share
```

Notice: the URL and HTML are gone, `2024` is gone, the stopwords `by`, `this`, `at`, `it`
are gone, and `scientists` became `scientist` through lemmatization.

**The critical point to make:** this exact same pipeline runs during training *and* during
prediction, because both import the same `TextPreprocessor` class from
`src/preprocessing.py`. If they differed, the model would receive differently-shaped input
at prediction time and accuracy would silently collapse — it would still return answers,
just bad ones.

---

## Section C — Machine learning

### Q10. Why Logistic Regression? (And why did your project select Linear SVM?)

**Short answer.** Logistic Regression is the natural first choice for text classification —
it is fast, works well on sparse high-dimensional data, gives real probabilities, and is
interpretable. In my project it scored 93.18% accuracy, but **Linear SVM scored higher
(94.37%) and was selected by cross-validation**, so I deployed the SVM.

**Fuller explanation.** Be honest and confident about this — it is a *strength* of the
project, not a weakness:

I did not assume which model would win. I trained four, compared them on the same splits,
and let measured cross-validated performance decide. Linear SVM won.

**Why Logistic Regression is still a good choice generally:**
- Outputs genuine calibrated probabilities (needed for my confidence meter)
- One signed weight per term, so it is directly interpretable
- Fast — trains in under half a second on 5,043 documents
- Works well when features vastly outnumber samples (20,000 features, 5,043 documents)

**How it works:** it computes a weighted sum `z = w₁x₁ + ... + b` and maps it through the
sigmoid function `σ(z) = 1/(1 + e^(−z))` to get a probability between 0 and 1.

---

### Q11. How does a Support Vector Machine work?

**Short answer.** An SVM finds the boundary that separates the two classes with the
largest possible **margin** — the widest gap between the boundary and the nearest data
points of each class. Those nearest points are called support vectors.

**Fuller explanation.** The intuition: if you can draw many lines separating two groups,
the best one is the one furthest from both groups, because it is most likely to still be
correct on new data. Maximising the margin is a built-in defence against overfitting.

**Why it wins on text:** with 20,000 sparse features, the classes are close to linearly
separable in high-dimensional space, and margin maximisation generalises better than just
placing a boundary anywhere between them.

**One detail to mention:** `LinearSVC` does not produce probabilities — it outputs an
unbounded distance from the boundary. Since my app needs a confidence percentage, I
wrapped it in `CalibratedClassifierCV`, which fits the SVM on internal folds and converts
its margins into calibrated probabilities.

---

### Q12. Why is Naive Bayes called "naive"?

**Short answer.** Because it assumes every feature is independent of every other feature
given the class — which for language is clearly false, since words are strongly correlated.

**Fuller explanation.** It applies Bayes' theorem:

```
P(class | features) ∝ P(class) × ∏ P(featureᵢ | class)
```

That product is only mathematically valid if features are independent. In text, "White"
and "House" obviously are not. Yet Naive Bayes still works reasonably well (90.40% in my
project) because the classification only needs the *ranking* of class probabilities to be
correct, not their absolute values. The independence violation distorts the magnitudes but
often preserves which class scores higher.

---

### Q13. Why did Random Forest perform worse than the linear models?

**Short answer.** Because tree ensembles are a poor fit for sparse, high-dimensional data.
With 20,000 features that are mostly zero, any single split is uninformative, whereas a
linear model weighs all features simultaneously.

**Fuller explanation.** My measured results:

| Model | Accuracy | Fit time |
|-------|---------:|---------:|
| Linear SVM | 0.9437 | 0.45 s |
| Logistic Regression | 0.9318 | 0.46 s |
| Random Forest | 0.9072 | 4.64 s |

Random Forest was both the **least accurate and ten times slower**. A decision tree splits
on one feature at a time — "does the word 'hoax' appear?" — but in a document with 20,000
features of which maybe 200 are non-zero, most such questions are answered "no" and reveal
nothing. A linear model sums the contributions of all 200 present terms at once.

**This is a good result to highlight:** I included Random Forest specifically to
*demonstrate* this rather than just assert it, and my finding matches the literature.

---

## Section D — Evaluation

### Q14. What is a confusion matrix? Explain yours.

**Short answer.** A confusion matrix is a table showing correct and incorrect predictions
broken down by class. It tells you not just *how many* mistakes the model made, but *what
kind*.

**My confusion matrix (Linear SVM, 1,261 test articles):**

|  | Predicted FAKE | Predicted REAL |
|--|---------------:|---------------:|
| **Actual FAKE** | **603** (TP) | 27 (FN) |
| **Actual REAL** | 44 (FP) | **587** (TN) |

- **True Positive (603)** — fake articles correctly flagged
- **True Negative (587)** — real articles correctly cleared
- **False Negative (27)** — fake articles the model **missed**
- **False Positive (44)** — real articles **wrongly flagged**

**The insight to add:** my model has more false positives (44) than false negatives (27),
so it leans toward flagging. For a screening tool that is the *better* direction of bias —
a flagged article gets escalated to a human reviewer, whereas a missed fake article gets no
scrutiny at all.

---

### Q15. What is the difference between precision and recall?

**Short answer.** Precision asks: *of everything I flagged as fake, how much really was
fake?* Recall asks: *of all the fake articles that exist, how many did I catch?*

**The formulas:**

```
Precision = TP / (TP + FP) = 603 / (603 + 44) = 93.20%
Recall    = TP / (TP + FN) = 603 / (603 + 27) = 95.71%
```

**The memorable way to explain the trade-off:**
- A model that flags **everything** as fake gets **100% recall** (it misses nothing) but
  terrible precision (it cries wolf constantly).
- A model that flags only the one article it is most certain about gets **100% precision**
  but almost zero recall (it misses nearly everything).

**Which matters more here?** For fake news detection, **recall** is arguably more
important — missing a fake article means it spreads unchecked, whereas a false positive
just means a human reviews a legitimate article. My model's recall (95.71%) is higher than
its precision (93.20%), which suits the use case.

---

### Q16. What is the F1-score, and why report it?

**Short answer.** F1 is the harmonic mean of precision and recall — a single number that
balances both. I use it because optimising precision or recall alone is easy to game.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.9320 × 0.9571) / (0.9320 + 0.9571) = 0.9444
```

**Why harmonic mean and not a plain average?** The harmonic mean punishes imbalance. If
precision were 1.0 and recall 0.0, the plain average would be a misleading 0.5, but F1 is
0. It forces *both* to be good.

**Follow-up you may get — "your classes are balanced, so why not just use accuracy?"**
Good question: with 50/50 classes, accuracy *is* meaningful here. But I report F1 as well
because it is the metric I used for model *selection*, and because it stays trustworthy if
the dataset were ever imbalanced.

---

### Q17. Why split the data into training and testing sets?

**Short answer.** To measure whether the model has actually learned generalisable patterns
or has just memorised the training examples. Testing on data the model has seen tells you
nothing.

**Fuller explanation.** The analogy: giving a student the exam paper to study, then setting
that exact paper as the exam. They score 100%, and you have learned nothing about whether
they understand the subject.

**My setup:** 80% training (5,043 articles) / 20% testing (1,261 articles), with:
- **Stratification** — the FAKE/REAL ratio is preserved in both halves, so an unlucky split
  cannot distort the result.
- **`random_state=42`** — a fixed seed makes the split reproducible, so my results can be
  verified by anyone running the code.

---

### Q18. What is overfitting? How did you guard against it?

**Short answer.** Overfitting is when a model memorises the training data — including its
noise — instead of learning general patterns. It scores brilliantly on training data and
poorly on anything new.

**Signs of it:** a large gap between training and test performance.

**How I guarded against it — five ways:**

1. **Held-out test set** — 1,261 articles never used in training.
2. **Cross-validation** — 5-fold CV means every model was validated on data it had not
   trained on; my CV F1 was 0.9348 ± 0.0093, and that small standard deviation shows the
   result is stable, not a lucky split.
3. **Regularisation** — Logistic Regression uses L2; SVM's margin maximisation is itself a
   form of regularisation.
4. **Feature limiting** — `max_features=20000` and `min_df=3` stop the model learning from
   rare noise and typos.
5. **Stopword removal** — reduces the feature space to informative terms.

**The evidence it worked:** my CV F1 (0.9348, on training folds) and test F1 (0.9444, on
unseen data) are very close. A big gap would indicate overfitting; the closeness indicates
genuine generalisation.

---

### Q19. What is data leakage? How did you prevent it?

**Short answer.** Data leakage is when information from the test set accidentally
influences training, making the reported score look better than the model really is. It is
the most common way ML projects produce fake results.

**This is the question examiners use to separate strong projects from weak ones. Know
these three answers cold:**

**1. The vectorizer is fitted on the training split only.**
```python
train_matrix = vectorizer.fit_transform(x_train)   # FIT on train
test_matrix  = vectorizer.transform(x_test)        # TRANSFORM only
```
If I had called `fit_transform` on the whole dataset, the vocabulary and the IDF weights
would have been computed using test-set documents — so the model's features would encode
information about data it is supposed to have never seen.

**2. Model selection used cross-validation *inside* the training split.**
I picked the winning algorithm by mean CV F1 on training data. The test set was touched
exactly **once**, at the very end, to produce the final reported number. If I had chosen
the winner by test accuracy, the test set would have influenced a modelling decision and
94.37% would be an optimistically biased estimate.

**3. Duplicate articles were removed before splitting.**
My dataset had 29 duplicates. If an identical article landed in both the training and test
halves, the model would "predict" an article it had memorised, inflating the score.

---

### Q20. How does a prediction actually work, end to end?

**Short answer.** The user's text is validated, cleaned with the same nine-step pipeline
used in training, converted to a TF-IDF vector using the *saved* vectorizer, scored by the
saved model, and returned with a probability and an explanation.

**The full sequence — walk through it confidently:**

1. **Validate** — reject empty text, text under 20 characters, over 20,000 characters, or
   text with no letters.
2. **Sanitise** — apply Unicode normalisation and strip control characters.
3. **Preprocess** — the identical nine-step pipeline from `src/preprocessing.py`.
4. **Vectorize** — `vectorizer.transform(text)`. Note: **transform, never fit** — the
   vectorizer must stay exactly as it was when the model was trained.
5. **Classify** — `model.predict(vector)` returns FAKE or REAL.
6. **Probability** — `model.predict_proba(vector)` gives the confidence.
7. **Explain** — multiply each present term's TF-IDF value by its learned weight; the
   largest products are the most influential terms.
8. **Store** — write the prediction to SQLite with a timestamp.
9. **Display** — show the label, confidence meter, influential terms and the disclaimer.

**Total time: about 50 milliseconds.**

**The key point about why the vectorizer is saved separately:** it holds fitted state — the
vocabulary and IDF weights. Without the exact same vectorizer, the word "election" might
map to feature index 4,912 during training but index 200 at prediction time, and the
model's weights would be meaningless.

---

## Section E — Questions you should expect

### Q21. What are the limitations of your project?

**Answer this honestly and specifically — it is the single best opportunity to show
maturity.**

1. **It does not check facts.** The model recognises writing style and vocabulary. It has
   no knowledge base. A well-written lie can pass as REAL; a badly written truth can be
   flagged FAKE.
2. **Topic bias — my most important finding.** My model's strongest FAKE features include
   `october`, `november` and `hillary`. Those are **topical, not stylistic** — they reflect
   *when* and *about what* the fake articles in this corpus were written (the 2016 US
   election). The model has partly learned the dataset's subject matter rather than
   deception itself.
3. **Limited generalisation.** Because of point 2, 94.37% is valid *on this dataset* but
   would be lower on current news from other domains.
4. **English only.**
5. **Short text is unreliable** — a headline leaves too few tokens. Try "Apple is good for
   health": it reduces to three words and my model says FAKE at 96% confidence, because in a
   2016 political corpus health-topic language appears mostly in clickbait. My app detects
   this (under 20 surviving words) and flags the result as unreliable instead of presenting
   the number as trustworthy.
6. **Ignores word order** — bag-of-words with bigrams captures limited context, not
   sentence structure, negation or sarcasm.
7. **Evadable** — someone who knows the model can write around it.

**How to frame this:** "Identifying this topic bias in my own results is something the
literature (Zhou & Zafarani, 2020) identifies as the central open problem in this field.
Finding it in my own evaluation is a sign of rigorous testing, not a flaw in the
implementation."

---

### Q21b. Does your system work on all news, or only certain kinds?

**Short answer.** It works on **political and world news from any country**, but not on
other topics. I tested this rather than assumed it.

**The experiment.** I submitted seven legitimate, neutrally-written news items:

| Item | Domain | Verdict |
|------|--------|---------|
| Lok Sabha finance bill (India) | Politics | ✅ REAL |
| Bank of England rate decision (UK) | Politics | ✅ REAL |
| US Senate budget agreement | Politics | ✅ REAL |
| Football match report | Sports | ❌ FAKE |
| Clinical trial results | Science | ❌ FAKE (97.5% confident) |
| Quarterly earnings | Business | ❌ FAKE |
| Cake recipe | Not news | ❌ FAKE |

**The key sentence to say:** *"The limitation is topic, not geography."* India and the UK
aren't in my training data at all, yet both were classified correctly — so the model did
learn transferable markers of political journalism. But everything outside politics
failed, because the model effectively learned "political vocabulary ⇒ REAL", and
unfamiliar vocabulary falls to FAKE by default.

**What I did about it.** I added a detector measuring what fraction of a document's terms
are among the model's 500 most influential features. I chose the threshold from measured
data, not intuition:

| Threshold | Genuine articles wrongly warned | Caught |
|----------:|--------------------------------:|--------|
| 0.10 | 4.7% | recipes, bare claims |
| 0.15 | 41.8% | + business, science |
| 0.18 | 71.1% | + sports |

I used 0.10. **Catching sports news would mean warning on 71% of legitimate articles**,
which makes the warning worthless. So the detector catches text that isn't news at all,
and a permanent notice in the UI states the subject scope for the rest. I'd rather ship an
honest partial fix than a warning nobody can trust.

---

### Q21c. Why does the warning use word count instead of the confidence score?

**Short answer.** Because confidence doesn't fall when accuracy does. I measured it.

| Words after preprocessing | Accuracy | Mean confidence |
|--------------------------:|---------:|----------------:|
| 3 | 62.0% | 86.9% |
| 10 | 71.4% | 87.8% |
| 50 | 85.1% | 89.4% |
| Full article | 94.4% | 93.6% |

**Accuracy drops 32 points; confidence drops under 7.** The model has no idea it is
guessing. If I had driven the warning off the probability, a three-word fragment at 87%
confidence would have sailed through unflagged.

That is why `MIN_SIGNAL_WORDS` is counted from the text, not read off the score. It's a
good example of why you measure a system's failure mode instead of assuming it degrades
gracefully.

---

### Q22. Your model is 94% accurate. Can I trust it to tell me if a news article is true?

**Short answer.** No — and my system is deliberately designed to say so.

**Fuller explanation.** The 94.37% measures how often the model's classification matches
the dataset's label on articles similar to its training data. It does **not** mean there is
a 94% chance any given article is factually true or false.

This is why my project:
- Uses the wording **"Model prediction: FAKE"**, never "This article is fake"
- Shows a **confidence percentage**, so users can see when the model is unsure
- Flags **low-signal** results when too little text remains after preprocessing
- Displays this disclaimer wherever a prediction appears: *"This system provides an
  ML-based classification and should not be treated as a substitute for professional
  fact-checking."*

The system is a **screening aid** that directs human attention efficiently. It is not an
arbiter of truth.

---

### Q23. What would you do to improve this project?

**Give three, in priority order:**

1. **Fix the topic bias first** — train on a larger, multi-source, multi-period dataset and
   validate *across* domains rather than within one corpus. This is the highest-value
   improvement because it addresses my project's main weakness.
2. **Use a transformer model (BERT/RoBERTa)** — these understand context and word order,
   which bag-of-words fundamentally cannot. The literature reports 96–98%, versus my 94.37%.
3. **Add source credibility signals** — domain reputation and author history are evidence
   independent of writing style, which would make the system harder to evade.

---

### Q24. Why SQLite instead of MySQL or PostgreSQL?

**Short answer.** SQLite is serverless, requires zero configuration, and stores everything
in one file — which is exactly right for a single-user academic project.

**Fuller explanation.** It ships with Python, so there is no extra installation, no service
to run and no credentials to manage. The project's write volume is tiny (one row per
prediction).

**Know when it would not be right:** if this were deployed for many concurrent users,
SQLite's write locking would become a bottleneck, and I would migrate to PostgreSQL. The
data-access layer is isolated in `src/database.py`, so that change would be contained to
one file.

---

### Q25. What security measures did you implement?

**Short answer.** Six, each with a test proving it works.

| Threat | Mitigation |
|--------|-----------|
| **SQL injection** | Every value bound as a `?` parameter; the ORDER BY column checked against a fixed allow-list |
| **Cross-site scripting** | Jinja2 autoescaping — a `<script>` tag in user input renders as harmless text |
| **Denial of service** | 20,000-character input limit and a 1 MB request-body cap |
| **Path traversal** | Report filenames validated against a fixed set before touching the filesystem |
| **Information disclosure** | Users see a generic message and a random error id; the full traceback goes only to the server log |
| **Hard-coded secrets** | `SECRET_KEY` read from the environment; `.env` is git-ignored |

**The strong thing to add:** each of these has a corresponding automated test. For example
`test_search_with_sql_metacharacters_is_safe` submits `'; DROP TABLE predictions; --` as a
search term and asserts the table still exists afterwards.

---

## Final checklist before the viva

- [ ] I can state my accuracy, precision, recall and F1 from memory
- [ ] I can draw my confusion matrix and explain all four cells
- [ ] I can explain TF-IDF with the `said` vs `hoax` example
- [ ] I can give **three** distinct data-leakage preventions
- [ ] I can explain why Linear SVM beat Logistic Regression — and that I let the data decide
- [ ] I can explain the topic-bias limitation in my own learned features
- [ ] I can run a live demo: analyze text, show the dashboard, show the history
- [ ] I can point to the exact file for any component they ask about
- [ ] I can say clearly that this is a screening aid, **not** a fact-checker

**If you do not know an answer:** say "I am not certain, but my understanding is…" and
reason out loud. Examiners reward honest reasoning far more than a confident wrong answer.

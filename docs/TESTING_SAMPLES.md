# Testing Guide & Demo Samples

Ready-to-paste text for demonstrating the system, with the result each one
actually produces.

> **Every sample below was run through the deployed model and the output
> recorded.** The verdicts and percentages are measured, not predicted. They will
> reproduce exactly as long as the model is not retrained.

---

## 1. What the model is actually judging

Before demonstrating, be clear about the basis — this is the question examiners
ask, and the honest answer is the strong one.

The model does **not** check facts. It has no knowledge base and cannot verify
whether an event happened. What it does is compare your text against writing
patterns learned from 6,304 labelled news articles.

It has learned, roughly, that:

| Signal of REAL | Signal of FAKE |
|----------------|----------------|
| Attribution: "officials **said**", "according to" | Urgency: "BREAKING", "SHARE before they delete" |
| Named sources, institutions, specific figures | Vague sources: "insiders", "they don't want you to know" |
| Neutral, measured tone | Emotive tone, heavy capitalisation |
| Procedural political vocabulary | Reader-directed commands: "Wake up", "Click here" |

`said` is the single strongest REAL signal in the trained model. That reflects a
genuine journalistic convention: professional reporting *attributes* claims,
whereas fabricated content simply asserts them.

### The two hard requirements

**1. It must be political or world news.** The training corpus is political news.
Political reporting from any country works — India and the UK both classify
correctly despite being absent from the training data. Sports, health, science,
business and entertainment do **not** work.

**2. It must be a real chunk of text.** Paste a paragraph or more. Very short
input leaves too few words after preprocessing for a meaningful result.

---

## 2. Samples that return REAL

Copy any of these whole.

### R1 — Agency-style political report → **REAL, 97.3%**

```
WASHINGTON (Reuters) - The Senate on Tuesday approved a bipartisan budget agreement, voting 71 to 28 to extend federal funding through the end of the fiscal year. The measure now moves to the House of Representatives, where leaders said they expect a vote before the end of the week. Senator Patty Murray, who helped negotiate the deal, said the agreement would provide certainty for federal agencies that have operated under short-term funding for months. A senior administration official, speaking on condition of anonymity, said the president was prepared to sign the bill once it reaches his desk. The Congressional Budget Office estimated the package would add $32 billion to discretionary spending over two years.
```

*Why it works:* attribution ("said" three times), named institutions, specific
figures, neutral tone.

### R2 — Campaign reporting → **REAL, 99.1%**

```
The candidate addressed supporters at a rally in Ohio on Thursday evening, outlining a policy platform focused on manufacturing jobs and infrastructure investment. Campaign officials said the event drew roughly 4,000 attendees. In remarks lasting about 40 minutes, the candidate criticised the opposing party's record on trade while avoiding direct attacks on individual opponents. A spokesperson for the campaign said additional stops in Pennsylvania and Michigan were planned for next week. Recent polling conducted by an independent research organisation showed the race remaining within the margin of error.
```

*Note:* this is the highest-confidence REAL sample. Useful as your opening demo.

### R3 — Government policy story → **REAL, 77.1%**

```
The State Department said on Monday that it would review its visa processing procedures following a report from the inspector general that identified delays averaging several months. The report recommended additional staffing and a modernised case management system. A department spokesperson said officials were reviewing the recommendations and would respond formally within 60 days. Members of the congressional oversight committee said they intended to hold hearings on the matter later this year. The review covers applications filed since the beginning of the previous fiscal year.
```

*Why this one is useful:* at 77.1% it is correct but **not** highly confident.
Good for showing that the confidence meter is meaningful rather than always
pinned at 99%.

---

## 3. Samples that return FAKE

### F1 — Conspiracy clickbait → **FAKE, 100.0%**

```
BREAKING: The mainstream media WON'T TELL YOU THIS! Government insiders have finally confirmed what patriots have been saying all along. The establishment elites are in total panic as this shocking truth spreads across the internet like wildfire. SHARE this article before they delete it forever! Wake up people - they have been lying to you for decades and now the evidence is finally out in the open. Click here to read the full story that the corrupt politicians and their media allies tried desperately to bury from the American public.
```

*Why it works:* urgency, sharing instruction, vague sourcing, no attribution.
The **Influential terms** panel is worth showing here.

### F2 — Sensational political → **FAKE, 100.0%**

```
You won't believe what was just discovered in leaked documents that expose the entire corrupt system. Sources close to the investigation revealed that everything we were told was a complete lie designed to deceive the American people. The liberal media is refusing to cover this bombshell story because it destroys their narrative completely. Share this everywhere before it gets censored! The truth is finally coming out and the establishment is terrified of what happens next when the people learn what really occurred behind closed doors.
```

### F3 — Anonymous-whistleblower hoax → **FAKE, 99.9%**

```
SHOCKING REVELATION: Anonymous whistleblower exposes the secret plan that they don't want you to know about. This explosive information has been suppressed by the corrupt establishment for years. Patriots everywhere are sharing this before the censors take it down. The evidence is undeniable and the mainstream media refuses to report on it. Wake up America - the truth about what these people have been doing will shock you to your core. Please share this important message with everyone you know immediately.
```

---

## 4. Samples that demonstrate the limitations

**Do not hide these — demonstrate them deliberately.** Showing that you know
where your system fails is worth more marks than pretending it always works.

### E1 — Short claim → **FAKE 96.6%, flagged unreliable**

```
Apple is Good for health
```

Only **3 words** survive preprocessing (`apple good health`). Triggers **both**
warnings: low-signal and out-of-domain. The verdict is greyed out and the
confidence reads "not reliable".

*What to say:* "This is correct behaviour, not a bug. Health vocabulary is
outside the training domain, and three words carry no signal. The system detects
both conditions and refuses to present the result as trustworthy."

### E2 — Recipe → **FAKE 96.2%, flagged out-of-domain**

```
Preheat the oven to 180 degrees and grease a round cake tin. Cream the butter and sugar together until pale, then beat in the eggs one at a time. Fold in the flour and bake for 35 minutes until a skewer comes out clean.
```

Domain ratio **0.000** — not one word carries learned weight. Clean demonstration
of the out-of-domain detector.

### E3 — Sports report → **FAKE 71.4%, NOT flagged** ⚠️

```
MANCHESTER - City secured a two-nil victory over their rivals on Saturday, with goals in the 23rd and 67th minutes. The manager said afterwards that the side had controlled the tempo throughout the match. The result moves the club three points clear at the top of the table with eight matches remaining.
```

**This one is wrong and the system does not catch it.** That is a known,
documented limitation, not an oversight.

*What to say:* "The detector can't flag this. I measured the trade-off: catching
sports news would mean warning on 71% of legitimate articles, which makes the
warning worthless. So I set the threshold to catch clearly non-news text, and
stated the subject-scope limitation in the interface instead."

### E4 — Below the minimum → **rejected before classification**

```
hi
```

Returns a validation error: *"That text is too short (2 characters). Please enter
at least 20 characters."* Demonstrates input validation.

### E5 — Headline only → **REAL 91.3%, NOT flagged** ⚠️

```
Senate approves federal budget agreement after long debate
```

Seven surviving words, above the `MIN_SIGNAL_WORDS = 5` threshold, so no warning
appears. It happens to be right here, but at this length measured accuracy is
only about **71%**. Raising the threshold would flag it; that is a tunable
trade-off in `src/config.py`.

---

## 5. Quick reference

| # | Sample | Expected | Confidence | Flags |
|---|--------|----------|-----------:|-------|
| R1 | Reuters budget report | REAL | 97.3% | — |
| R2 | Campaign rally | REAL | 99.1% | — |
| R3 | State Department review | REAL | 77.1% | — |
| F1 | Conspiracy clickbait | FAKE | 100.0% | — |
| F2 | Sensational political | FAKE | 100.0% | — |
| F3 | Whistleblower hoax | FAKE | 99.9% | — |
| E1 | "Apple is Good for health" | FAKE | 96.6% | low-signal + out-of-domain |
| E2 | Cake recipe | FAKE | 96.2% | out-of-domain |
| E3 | Sports report | FAKE (wrong) | 71.4% | none — known gap |
| E4 | "hi" | rejected | — | validation error |
| E5 | Headline only | REAL | 91.3% | none — soft spot |

---

## 6. Suggested demo sequence (about 5 minutes)

1. **R2** — a clean REAL at 99.1%. Establishes that it works.
2. **F1** — a clean FAKE at 100%. Open the **Influential terms** panel and point
   out `share`, `mainstream media`, `shocking`.
3. **R3** — REAL at 77.1%. Shows the confidence meter carries real information.
4. **E1** — "Apple is Good for health". Show both warnings firing.
5. **E3** — the sports report. Volunteer the failure and explain the measured
   trade-off behind it.
6. **Dashboard** — model comparison table, confusion matrix, per-class metrics.
7. **History** — mention it is stored in the browser, not on the server, so a
   shared link keeps each visitor's history private.

Finish on the disclaimer: *this is a screening aid, not a fact-checker.*

---

## 7. Finding your own test articles

**For REAL:** copy a body paragraph from Reuters, AP, BBC News or PTI — politics
or world news sections only. Skip the headline; paste the article body.

**For FAKE:** partisan blogs and viral political posts with heavy capitalisation
and sharing instructions work well. Search for phrases like *"they don't want you
to know"* or *"mainstream media won't report"*.

**Avoid:** sports, entertainment, health, technology and business — the model is
unreliable on all of them, regardless of whether the article is genuine.

---

## 8. If you retrain

`python src/train_model.py` is deterministic (`random_state = 42`), so identical
data reproduces identical numbers. But if you **change the dataset, the
preprocessing rules or the TF-IDF settings**, every figure in this document
becomes stale. Re-run these samples and update the tables rather than quoting
numbers the current model no longer produces.

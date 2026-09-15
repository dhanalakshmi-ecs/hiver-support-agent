\# AI Customer Support Agent — AmazonHelp



\## 1. Problem Framing



This project builds an AI customer-support agent for AmazonHelp using real customer-support conversations from Twitter.



The agent performs three tasks:



1\. Classifies the customer's message into a small set of support intents.

2\. Retrieves historically similar AmazonHelp conversations and uses them as evidence for drafting a reply.

3\. Decides whether the case can be auto-handled or should be escalated to a human.



\### What Good Means



For this project, a good support agent should:



\- identify the customer's main problem correctly;

\- retrieve relevant historical support examples;

\- produce a concise and useful reply;

\- avoid inventing account-specific information, refunds, dates, or policies;

\- escalate cases that are risky, urgent, unresolved, or explicitly require human help.



\### What I Chose Not to Build



I intentionally did not build:



\- real Amazon account integration;

\- real order/refund actions;

\- a web dashboard;

\- autonomous policy enforcement;

\- full-scale production deployment.



The goal was to build and evaluate a reproducible support-agent pipeline rather than a production customer-service platform.



\---



\## 2. Dataset and Brand Selection



The primary dataset is the Customer Support on Twitter dataset.



I selected the `AmazonHelp` brand because it contains a large number of customer-support interactions and clear customer-to-brand response pairs.



The original dataset contains millions of tweets. For this project, I extracted AmazonHelp interactions and constructed customer-message → AmazonHelp-reply pairs.



After cleaning:



\- AmazonHelp conversation pairs: 6,893

\- Clean usable conversations: 6,420

\- Training sample: 500

\- Golden evaluation set: 200



The raw Twitter dataset is intentionally excluded from Git because of its size.



\---



\## 3. Intent Taxonomy



I defined nine intents from the observed AmazonHelp conversations:



1\. `delivery\_issue`

2\. `order\_issue`

3\. `payment\_issue`

4\. `refund\_return`

5\. `prime\_issue`

6\. `account\_issue`

7\. `product\_device\_issue`

8\. `cancellation\_issue`

9\. `other`



A codebook containing definitions, inclusion/exclusion rules, and edge cases is included in `codebook.md`.



A key annotation rule is that the customer's main requested action determines the label. For example, refund/return requests are classified as `refund\_return`, while delivery-specific problems are classified as `delivery\_issue`.



\---



\## 4. System Architecture



The pipeline is:



Customer message

→ Intent classifier

→ Historical retrieval

→ Escalation decision

→ Response generation



\### Intent Classification



Two simple approaches were compared:



\- TF-IDF + Logistic Regression

\- SentenceTransformer embeddings + Logistic Regression



The semantic classifier performed better and was selected for the main pipeline.



Model:



`sentence-transformers/all-MiniLM-L6-v2`



Classifier:



`LogisticRegression(class\_weight="balanced")`



\### Historical Retrieval



Historical AmazonHelp conversations are embedded using the same MiniLM sentence encoder.



For a new customer message, the system retrieves the most similar historical customer-support cases using cosine similarity.



The top three cases are provided as evidence to the response generator.



\### Response Generation



The response generator uses an LLM and the retrieved historical cases.



The prompt instructs the model to:



\- answer the customer's actual issue;

\- use historical examples as grounding;

\- avoid inventing account/order information;

\- avoid claiming unsupported policies;

\- remain concise and professional.



\### Escalation



The escalation layer uses conservative rules.



Cases can be escalated when they contain:



\- security or payment risk;

\- account-related issues;

\- explicit requests for a human;

\- repeated or unresolved problems;

\- urgent situations;

\- serious delivery/product problems;

\- strong dissatisfaction;

\- low-confidence intent predictions.



\---



\## 5. Evaluation



\### Intent Baselines



I compared three approaches.



| Method | Accuracy | Macro F1 | Weighted F1 |

|---|---:|---:|---:|

| Majority baseline | 0.35 | 0.06 | 0.18 |

| TF-IDF + Logistic Regression | 0.37 | 0.23 | 0.38 |

| Semantic + Logistic Regression | 0.40 | 0.29 | 0.42 |



The semantic classifier was therefore used in the final pipeline.



\### Golden Evaluation



On the 200-example golden set:



\- Accuracy: 0.60

\- Macro F1: 0.52

\- Weighted F1: 0.62



The higher golden-set result compared with the held-out test result suggests that the golden set has a somewhat easier or different distribution. Therefore, I do not treat 60% accuracy as production performance.



\---



\## 6. Reply Quality Evaluation



I evaluated 30 generated replies using an LLM judge.



The judge scored five criteria from 1–5:



\- Correctness

\- Relevance

\- Grounding

\- Helpfulness

\- No Hallucination



Results:



| Criterion | Mean Score |

|---|---:|

| Correctness | 4.27 / 5 |

| Relevance | 4.83 / 5 |

| Grounding | 4.10 / 5 |

| Helpfulness | 4.33 / 5 |

| No Hallucination | 4.87 / 5 |

| Overall | 4.48 / 5 |



These results are encouraging but should not be interpreted as proof that every generated response is safe or correct.



\---



\## 7. Human Validation of the LLM Judge



I independently blind-rated six previously unseen generated replies using the same five criteria.



Human mean scores were:



\- Correctness: 4.50

\- Relevance: 4.67

\- Grounding: 3.67

\- Helpfulness: 4.17

\- No Hallucination: 4.50



Comparison with the blind LLM judge showed:



\- Correctness exact agreement: 50.0%, Cohen's κ = 0.182

\- Relevance exact agreement: 50.0%, Cohen's κ = 0.000

\- Grounding exact agreement: 16.7%, Cohen's κ = 0.000

\- Helpfulness exact agreement: 50.0%, Cohen's κ = 0.250

\- No Hallucination exact agreement: 66.7%, Cohen's κ = 0.250



This is a small sample, so the agreement statistics are unstable. More importantly, the weak grounding agreement shows that the LLM judge should be treated as a supplementary signal rather than ground truth.



\---



\## 8. Escalation Strategy



The escalation system is deliberately conservative.



AUTO-HANDLE is used only when the case does not trigger a high-risk or low-confidence rule.



ESCALATE is preferred for:



\- security/payment risks;

\- account problems;

\- explicit human requests;

\- repeated unresolved complaints;

\- urgent cases;

\- serious delivery/product failures;

\- strong dissatisfaction;

\- low-confidence predictions.



This reduces the risk of confidently producing an inappropriate automated response.



\---



\## 9. Top Failure Modes



\### 1. Other vs Delivery Confusion



Generic support messages can resemble delivery complaints.



Example pattern:



"I have been waiting for help with my order."



The message may contain order/delivery language without clearly describing a delivery problem.



\### 2. Sparse Minority Intents



Some intents such as account and cancellation have relatively few examples. Their classifier performance is therefore less reliable.



\### 3. Historical Retrieval Can Be Too Similar



If the exact customer message exists in the historical dataset, retrieval can return itself with similarity close to 1.0.



This can make grounding appear stronger than it would be on a genuinely unseen customer query.



\### 4. Historical Responses Are Not Guaranteed Policies



A historical AmazonHelp response shows what happened in a past conversation. It is not necessarily a current or universal Amazon policy.



\### 5. LLM Response Can Overgeneralize



A generated response may infer that a particular resolution is available even when the retrieved historical examples do not provide enough evidence.



\---



\## 10. What Is Misleading About My Headline Number?



The strongest headline result is the 60% accuracy on the 200-example golden set.



This number is useful, but it is misleading if interpreted as production accuracy.



Reasons:



1\. The golden set is small.

2\. The golden set distribution differs from the held-out test distribution.

3\. Some minority intents have very few examples.

4\. Historical retrieval can accidentally retrieve extremely similar or identical messages.

5\. The LLM-generated reply evaluation uses a small sample.

6\. LLM-judge agreement with human ratings is weak, especially for grounding.

7\. The system has not been tested in a live customer-support environment.



Therefore, the 60% golden accuracy should be viewed as an evaluation result on this specific sample, not as a production performance guarantee.



\---



\## 11. Limitations



The current system has several limitations:



\- limited labelled training data;

\- class imbalance;

\- small golden evaluation set;

\- simple rule-based escalation;

\- no real-time Amazon systems;

\- no live policy verification;

\- possible retrieval leakage;

\- limited human validation of the LLM judge.



\---



\## 12. What I Would Do With One More Week



With one additional week, I would prioritize:



1\. Increase the labelled training set.

2\. Add hard-negative retrieval examples.

3\. Prevent exact-message retrieval leakage during evaluation.

4\. Improve minority-intent classification.

5\. Calibrate intent confidence.

6\. Expand human evaluation to a larger blind sample.

7\. Improve grounding checks between generated claims and retrieved evidence.

8\. Add a stronger escalation classifier.

9\. Test more realistic unseen customer messages.



\---



\## 13. Repository Structure



```text

hiver-support-agent/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── golden/

│

├── evaluation/

│   └── majority\_baseline.py

│

├── outputs/

│

├── analyze\_amazon\_intents.py

├── analyze\_escalation\_thresholds.py

├── analyze\_predictions.py

├── analyze\_semantic\_predictions.py

├── apply\_escalation\_labels.py

├── baseline.py

├── calculate\_human\_agreement.py

├── check\_golden.py

├── clean\_amazon\_data.py

├── codebook.md

├── confusion\_analysis.py

├── create\_blind\_human\_samples.py

├── create\_human\_reply\_ratings.py

├── create\_labeling\_sample.py

├── create\_training\_sample.py

├── escalation.py

├── evaluate\_baselines\_on\_golden.py

├── evaluate\_blind\_reply\_quality.py

├── evaluate\_escalation.py

├── evaluate\_escalation\_human.py

├── evaluate\_intent.py

├── evaluate\_replies.py

├── evaluate\_reply\_quality.py

├── inspect\_data.py

├── intent\_predictor.py

├── pipeline.py

├── prepare\_amazon\_data.py

├── prepare\_escalation\_sample.py

├── repair\_reply\_samples.py

├── requirements.txt

├── response\_generator.py

├── retriever.py

├── semantic\_classifier.py

└── tfidf\_classifier.py


\# Decision Log — AmazonHelp AI Customer Support Agent



\## 1. Selected AmazonHelp as the brand

I selected AmazonHelp because it has a sufficiently large set of customer-support interactions in the dataset, allowing both intent classification and historical-response retrieval.



\## 2. Built customer-to-Amazon conversation pairs

I paired customer tweets with AmazonHelp replies using the response relationship in the dataset. This preserves the actual support context rather than treating tweets as isolated text.



\## 3. Created a custom nine-intent taxonomy

I defined nine intents from the observed AmazonHelp conversations:

\- delivery\_issue

\- order\_issue

\- payment\_issue

\- refund\_return

\- prime\_issue

\- account\_issue

\- product\_device\_issue

\- cancellation\_issue

\- other



This keeps the classification problem small and practical.



\## 4. Created an explicit annotation codebook

I documented definitions, inclusion/exclusion rules, and edge cases in `codebook.md` so that labeling decisions were consistent.



\## 5. Added precedence rules for ambiguous cases

Delivery-related issues take precedence over generic order issues, while refund/return takes precedence when the main customer request is returning an item or receiving money back.



\## 6. Kept training and golden evaluation data separate

I created a 500-example labeled training sample and a separate 200-example golden evaluation set. This prevents the same examples from being used for both model development and final evaluation.



\## 7. Compared simple and semantic classifiers

I evaluated a majority baseline, TF-IDF + Logistic Regression, and sentence-embedding + Logistic Regression. This provides evidence that the semantic approach improves over simple baselines.



\## 8. Used class weighting

The intent distribution is highly imbalanced, so the classifiers use class weighting to reduce the tendency to predict only frequent intents.



\## 9. Used historical AmazonHelp conversations for retrieval

Instead of generating replies from general knowledge alone, the system retrieves similar historical AmazonHelp conversations and uses them as evidence for response generation.



\## 10. Retrieved the top three similar cases

I selected the top three historical examples to give the response generator enough context without overwhelming the prompt.



\## 11. Added anti-hallucination instructions to response generation

The response generator is instructed not to invent order information, refund decisions, delivery dates, account details, or policies that are not supported by the retrieved evidence.



\## 12. Used conservative escalation rules

The system escalates high-risk security/payment issues, account issues, explicit human requests, repeated/unresolved complaints, urgent cases, serious delivery/product problems, strong dissatisfaction, and low-confidence predictions.



\## 13. Did not implement real account actions

The prototype only drafts responses and makes escalation decisions. It does not access customer accounts, issue refunds, modify orders, or perform other irreversible actions.



\## 14. Blindly validated the LLM judge with human ratings

I created a separate blind sample and independently rated generated replies using the same five criteria as the LLM judge. The comparison showed that the judge should be treated as a supplementary signal rather than ground truth.



\## 15. Excluded raw data and secrets from Git

The large raw Twitter dataset and environment files containing API secrets are excluded through `.gitignore`. This keeps the repository safer and avoids committing credentials or unnecessary large files.


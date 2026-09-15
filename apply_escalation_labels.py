import pandas as pd

path = "data/golden/escalation_sample.csv"

df = pd.read_csv(path, dtype=str)
labels = {
    "69804": ("ESCALATE", "Customer has not received expected communication"),
    "86126": ("ESCALATE", "Repeated delivery failure and strong frustration"),
    "26880": ("ESCALATE", "Unauthorized recurring Prime charge"),
    "43637": ("ESCALATE", "Delivery problem with complaint about service"),
    "96850": ("ESCALATE", "Package mishandled by driver"),
    "36348": ("ESCALATE", "Package damaged by improper delivery"),
    "28431": ("ESCALATE", "Issue unresolved after contacting support multiple times"),
    "118838": ("ESCALATE", "Package damaged by rain due to delivery handling"),
    "24955": ("ESCALATE", "Scheduled replacement collection was missed"),
    "108153": ("AUTO-HANDLE", "General conversation"),
    "21904": ("AUTO-HANDLE", "Routine product information"),
    "108248": ("AUTO-HANDLE", "General conversation"),
    "64166": ("ESCALATE", "Support promised callback but failed to respond"),
    "29540": ("ESCALATE", "Refund/replacement unresolved"),
    "69702": ("ESCALATE", "Package potentially damaged after improper delivery"),
    "76061": ("ESCALATE", "Customer requests immediate resolution"),
    "36527": ("ESCALATE", "Repeated unresolved replacement issue"),
    "13919": ("AUTO-HANDLE", "Routine delivery-status question"),
    "126085": ("ESCALATE", "Existing escalated case still unresolved"),
    "9810": ("ESCALATE", "Return pickup failed and refund requested"),
    "51592": ("AUTO-HANDLE", "Routine delivery-status question"),
    "69806": ("ESCALATE", "Customer disputes product/refund resolution"),
    "76039": ("AUTO-HANDLE", "Routine delivery tracking question"),
    "22472": ("ESCALATE", "Customer says previous support process failed"),
    "113742": ("AUTO-HANDLE", "Routine Prime delivery-delay issue"),
    "65787": ("ESCALATE", "Repeated attempts and severe frustration"),
    "28939": ("ESCALATE", "Severe delivery delay with unresolved support"),
    "91238": ("AUTO-HANDLE", "Simple acknowledgement"),
    "35909": ("ESCALATE", "Customer demands immediate delivery resolution"),
    "122002": ("ESCALATE", "Strong dissatisfaction with customer service"),
    "5137": ("AUTO-HANDLE", "Routine missing-package query"),
    "21551": ("ESCALATE", "Highly hostile interaction requiring human handling"),
    "36539": ("ESCALATE", "Serious complaint about support"),
    "69727": ("ESCALATE", "Prime delivery failure requiring account/order investigation"),
    "46561": ("ESCALATE", "Wrong product edition and repeated failure"),
    "99933": ("ESCALATE", "Account verification issue with no response"),
    "69969": ("ESCALATE", "Customer reports previous false promises"),
    "114373": ("AUTO-HANDLE", "General product/service information"),
    "108076": ("AUTO-HANDLE", "Routine device troubleshooting follow-up"),
    "16877": ("AUTO-HANDLE", "General review-related information"),
    "26975": ("ESCALATE", "Explicit request for a phone call"),
    "693": ("AUTO-HANDLE", "Routine Prime benefits question"),
    "22172": ("AUTO-HANDLE", "Routine refund status"),
    "13558": ("AUTO-HANDLE", "Routine delivery-status question"),
    "3751": ("AUTO-HANDLE", "Simple product clarification"),
    "14298": ("AUTO-HANDLE", "General delivery complaint without specific intervention"),
    "115931": ("ESCALATE", "Cancellation failed and requires support action"),
    "91494": ("AUTO-HANDLE", "General seller pricing question"),
    "59550": ("AUTO-HANDLE", "Shipping-date information"),
    "65869": ("ESCALATE", "Return pickup failure and fraud allegation")
}

df["customer_tweet_id"] = df["customer_tweet_id"].astype(str)

for tweet_id, (label, note) in labels.items():
    mask = df["customer_tweet_id"] == tweet_id
    df.loc[mask, "escalation_label"] = label
    df.loc[mask, "escalation_notes"] = note

# Check whether anything is still empty
missing = (
    df["escalation_label"].isna()
    | (df["escalation_label"].astype(str).str.strip() == "")
)

print("Missing labels:", missing.sum())
print("\nLabel counts:")
print(df["escalation_label"].value_counts())

df.to_csv(path, index=False)

print("\nEscalation labels filled successfully!")
print("File updated:", path)
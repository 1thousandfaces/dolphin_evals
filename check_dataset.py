
from collections import Counter
from datasets import load_dataset

# load dataset
ds = load_dataset("sxm4/Refusals-V5-Eval", "default", split="train")

# count conversation lengths
counts = Counter()
for row in ds:
    convo = row.get("conversation") or row.get("conversations") or row.get("turns") or []
    counts[len(convo)] += 1

# pretty-print histogram sorted by length
print("conversation_length,count")
for length, n in sorted(counts.items()):
    print(f"{length},{n}")

# optional: quick summary
total = sum(counts.values())
avg_len = sum(k*v for k,v in counts.items()) / total
print(f"\nTotal conversations: {total}")
print(f"Average length: {avg_len:.2f}")
print(f"Min length: {min(counts)}, Max length: {max(counts)}")
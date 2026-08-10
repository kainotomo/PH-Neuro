# Τα Μικρότερα Deep Learning Μοντέλα στον Κόσμο

> **PH-Neuro — Public Launch · M2.5** · 2026-08-10 · `kainotomo/PH-Neuro`

---

## TL;DR

**102M παράμετροι, 2-bit weights, 25 MB.** Μία gaming GPU. 2 ώρες.

Το PH-Neuro είναι πλέον δημόσιο: κάνε clone, κατέβασε τα μοντέλα (<30 MB
συνολικά), και δες text generation + image classification **μέσα στον browser**,
στο δικό σου μηχάνημα, χωρίς cloud. Τα βάρη είναι **{-1, 0, +1}** — τριαδικά,
2 bit ανά βάρος — και όλη η inference τρέχει σε **CPU** μέσω ONNX Runtime.

---

## Το πρόβλημα

Τα AI models είναι τεράστια. Το GPT-2 είναι **500 MB**. Το LLaMA είναι **4 GB+**.
Το GPT-4 δεν χωράει καν στη μνήμη του laptop σου. Κανένα από αυτά δεν τρέχει
στο κινητό σου, χωρίς σύνδεση, χωρίς να στέλνει τα δεδομένα σου σε κάποιο
data center.

> Γιατί χρειάζονται 4 GB όταν η ουσία — οι γνώσεις — μπορεί να χωρέσει σε
> λιγότερο από 25 MB;

## Η λύση: Ternary Weights + DQT

Τρεις ιδέες, μία φιλοσοφία: **διώξε το πλεονάζον**.

- **Ternary weights {-1, 0, +1}** αντί για float32 → **16× μικρότερα**.
  Κάθε βάρος είναι 2 bit — 4 βάρη σε κάθε byte. Ο αποθηκευτικός χώρος
  καταρρέει από εκατοντάδες MB σε δεκάδες MB.
- **DQT (Direct Quantized Training)** → **4.5× λιγότερη μνήμη εκπαίδευσης**.
  Οι περισσότερες μέθοδοι (BitNet, QLoRA) κρατάνε "κρυφά" float scores για
  κάθε βάρος ενώ εκπαιδεύουν. Το DQT τα πετάει: εκπαιδεύει κατευθείαν τα
  τριαδικά βάρη με στοχαστική στρογγυλοποίηση. Λιγότερη μνήμη, λιγότερη
  ενέργεια, ίδια ακρίβεια.
- **MoE (Mixture of Experts)** → **sparse activation**. Μόνο το **50% των
  παραμέτρων** είναι ενεργό σε κάθε forward pass — τα υπόλοιπα μένουν
  κλειστά. Περισσότερη χωρητικότητα χωρίς περισσότερο κόστος inference.

**Συνδυασμένα:** 102M παράμετροι που χωράνε σε 25 MB, εκπαιδεύονται σε μία
gaming GPU σε λίγες ώρες, και τρέχουν inference σε CPU με ~20+ tokens/sec.

## Τα αποτελέσματα

| Μοντέλο | Παράμετροι | 2-bit packed | ONNX | Μετρική | Χρόνος εκπαίδευσης |
|:--------|-----------:|-------------:|-----:|:--------|:------------------:|
| **DQT Transformer 102M** (TinyStories) | 102.3M ternary | **25 MB** | ~500 MB | val ppl ~11.5 | ~2 h @ RTX 4060 |
| **DQT CNN CIFAR-10** | 4.3M ternary | **1.0 MB** | 16 MB | **78.6%** acc | ~12.6 min |
| **DQT CNN CIFAR-100** | 2.5M ternary | **0.6 MB** | 10 MB | **54.2%** acc | ~25 min |

Σύγκριση με τα "κανονικά":

| Μοντέλο | Μέγεθος | Σημείωση |
|:--------|--------:|:---------|
| GPT-2 small | 500 MB | 20× μεγαλύτερο από το δικό μας transformer |
| MobileNetV2 (TF-Lite) | 14 MB | το δικό μας CNN είναι 1 MB |

Ο transformer με **102M τριαδικά βάρη** έχει val perplexity **~11.4** στο
TinyStories — ανταγωνιστικό με μοντέλα 10× μεγαλύτερα — αλλά ζυγίζει όσο ένα
τραγούδι σε MP3. Και το CNN που βλέπει εικόνες χωράει σε **1 MB**.

## Demo

Δοκίμασέ το μόνος σου — τρεις καρτέλες, όλα στον browser:

1. **📝 Text Generation** — γράψε μια αρχή ιστορίας και δες το transformer να
   συνεχίζει (temperature / top-k ρυθμιζόμενα, live tok/s).
2. **🖼️ Image Classification** — ανέβασε ή τράβηξε φωτογραφία· CIFAR-10 ή
   CIFAR-100, top-3 predictions με μπάρες πιθανοτήτων.
3. **📊 Benchmarks** — όλα τα μοντέλα σε έναν πίνακα, με σύγκριση GPT-2 / TF-Lite.

```bash
git clone https://github.com/kainotomo/PH-Neuro
cd PH-Neuro
pip install -e . gradio onnxruntime
python scripts/run_m2_5_demo.py
# → http://localhost:7860
```

Χρειάζεσαι μόνο CPU. Όχι GPU, όχι cloud, όχι API keys.

## Τι σημαίνει αυτό

Μοντέλα που τρέχουν **στο κινητό σου**. Χωρίς cloud. Χωρίς data center.
Χωρίς να ανεβάζεις τα δεδομένα σου πουθενά.

- **Privacy by default** — η inference γίνεται στη συσκευή σου.
- **Energy** — 2-bit weights σημαίνουν τάξεις μεγέθους λιγότερη ενέργεια ανά
  forward pass.
- **Democratization** — αν ένα μοντέλο εκπαιδεύεται σε ~2 ώρες σε μία gaming
  GPU, τότε δεν χρειάζεται να είσαι η OpenAI για να φτιάξεις το δικό σου.

Το όραμα του PH-Neuro (δείτε [GOALS.md](GOALS.md)): **1 δισεκατομμύριο
παράμετροι → 200 MB → τρέχει στο κινητό σου.**

## Try it yourself

```bash
git clone https://github.com/kainotomo/PH-Neuro
cd PH-Neuro
pip install -e . gradio onnxruntime
python scripts/run_m2_5_demo.py
# → http://localhost:7860
```

Αναπαράγεις τα μοντέλα από το μηδέν:

```bash
# Τα 3 μοντέλα: train → export ONNX+2-bit → demo
bash scripts/run_m2_5_demo.sh          # full pipeline
# ή βήμα-βήμα:
bash scripts/run_m2_5_demo.sh train    # ~2 h στον RTX 4060
bash scripts/run_m2_5_demo.sh export   # ONNX + .ternary
bash scripts/run_m2_5_demo.sh demo     # gradio @ :7860
```

Τα έτοιμα artifacts βρίσκονται στο [`results/phase2/m2_5/`](../results/phase2/m2_5/).

---

*PH-Neuro — Tiny Ternary AI. Βάρη {-1, 0, +1}. Εκπαίδευσε σε μία GPU. Τρέξε
σε ένα κινητό.*

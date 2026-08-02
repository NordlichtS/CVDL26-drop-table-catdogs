### Suggested eight-page allocation

| Content | Approximate space |
|---|---:|
| Abstract | 0.25 page |
| Introduction | 0.9 page |
| Related Work | 0.8 page |
| Method | 2.0 pages |
| Experiments and XAI | 3.3 pages |
| Conclusions | 0.5 page |
| Appendix contributions | outside or at end as allowed |





## 5. Conclusions

We developed a two-stage system for fine-grained recognition of ten cat and ten dog categories with an explicit reject output. The final code combines a COCO-pretrained YOLOv6 detector with separate from-scratch EfficientNetV2-S classifiers. Detection encodes the largest-animal policy and reduces classifier input to a standardized crop; species routing reduces each classification problem to ten classes. The project also exposed practical limitations in data provenance, class imbalance, ambiguous labels, preprocessing consistency, and the risk of contextual shortcut learning.

**TODO: Replace the next sentence after final evaluation.** Under the locked evaluation protocol, the system achieved **[accuracy]**, **[macro F1]**, and **[reject recall]**, with **[mean +/- standard deviation]** seconds warm end-to-end latency on **[hardware]**. Controlled ablations showed that **[main supported finding]**, while **[negative result]** did not improve performance. Grad-CAM examples indicated **[carefully qualified observation]**, but also revealed **[failure/counterexample]**.

The main limitation is **[choose based on evidence: detector dependence, data ambiguity, imbalance, absence of confidence calibration, or domain shift]**. Future work should follow directly from this limitation, for example detector calibration, explicit out-of-distribution rejection, better duplicate-aware data curation, or evaluation on a broader domain-shift set. Do not introduce Ranger, adaptive padding, or another untested method here as though it were part of the completed project.

What this section must contain:

- Direct answers to the research questions.
- The main quantitative result and the most important limitation.
- No new experiments or citations unless unavoidable.
- A short, evidence-based future-work statement.

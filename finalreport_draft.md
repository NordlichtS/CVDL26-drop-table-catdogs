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


## 2. Related Work

### 2.1 Fine-grained animal recognition and datasets

The Oxford-IIIT Pet dataset contains 37 cat and dog categories and supplies breed labels together with localization-oriented annotations, making it a standard reference for fine-grained pet recognition [@parkhi2012catsdogs]. Stanford Dogs contains 20,580 images across 120 dog breeds and was constructed from ImageNet for fine-grained categorization [@khosla2011stanforddogs; @deng2009imagenet]. These datasets motivate both the use of localized animal regions and careful evaluation of visually similar classes. Our assembled dataset reuses only the challenge-relevant classes and supplements missing or underrepresented categories through additional sources [C6]. **[TODO: Replace this general statement with the final manifest's exact source-by-class counts.]**

### 2.2 Efficient object detection

YOLO-style detectors perform localization and classification in a single stage. YOLOv6 targets efficient industrial deployment and provides model scales with different speed-accuracy tradeoffs [@li2022yolov6]. We use YOLOv6-S with weights pretrained on COCO, whose object vocabulary includes cat and dog classes [@lin2014coco]. Unlike work that trains a task-specific detector, our detector is used off the shelf as a localization and routing component. **[TODO: State the exact YOLOv6 checkpoint/release and license in the final reproducibility appendix.]**

### 2.3 Efficient image classification

EfficientNetV2 combines training-aware architecture search and scaling to improve training speed and parameter efficiency [@tan2021efficientnetv2]. EfficientNetV2-S was selected as the breed-classification backbone. The project explored both ImageNet initialization and random initialization [S3, slides 3-5]. The shipped `AnimalClassifier`, however, constructs the backbone with `weights=None`; any claim about ImageNet-pretrained performance is therefore an experimental baseline rather than the final model [C2].

### 2.4 Explainable AI for visual classifiers

Grad-CAM uses gradients flowing into a convolutional layer to form a coarse class-specific localization map [@selvaraju2017gradcam]. We use it as a qualitative diagnostic for contextual shortcut learning and localized breed cues [C5]. Grad-CAM does not establish causal feature use, so the analysis must include counterexamples and should avoid treating a heatmap as proof that a model has learned anatomy.

What this section must contain:

- Directly relevant literature, not a catalogue of every architecture considered.
- Original papers for YOLOv6, EfficientNetV2, Grad-CAM, COCO, ImageNet, Oxford-IIIT Pet, and Stanford Dogs.
- A clear contrast between prior work and this project's use of off-the-shelf detection plus task-specific classification.
- **TODO:** Add an academically appropriate source only if SpeciesNet or MegaDetector remains part of a documented negative experiment; otherwise omit them from the final paper.

## 3. Method

### 3.1 Problem formulation

Let an input image be (x). The required output is (y \in \{-1, 0, \ldots, 19\}), where labels 0-9 denote cat classes, labels 10-19 denote dog classes, and `-1` denotes rejection [C4]. The detector produces a set of detections (D=\{(b_i,c_i,s_i)\}), where (b_i=(x_{1i},y_{1i},x_{2i},y_{2i})), (c_i) is a COCO class, and (s_i) is the detector score. From detections classified as cat or dog, the current implementation chooses

\[
i^* = \arg\max_i (x_{2i}-x_{1i})(y_{2i}-y_{1i}).
\]

If no valid cat or dog box remains, the system returns `-1`. Otherwise, the detected species selects a ten-class classifier. A local dog prediction is shifted by 10 to recover the global challenge label [C1, C4].

### 3.2 Detection, masking, and crop construction

The detector uses YOLOv6-S loaded through `torch.hub` with a local checkpoint [C1]. Input images are converted from OpenCV BGR to RGB, resized to 640 x 640 for detector inference, and evaluated for COCO cat ID 15 and dog ID 16. The largest valid target box is retained. The current code then sets pixels inside every other returned detection box to zero before extracting the primary crop. Coordinates are clamped to image bounds, and the crop is bilinearly resized to 224 x 224 [C1].

This description intentionally excludes two claims found in presentation drafts: adaptive padding around the selected box and a custom detector-confidence threshold. Neither is implemented in the current detector or router. They may be added only if evaluated and included in the frozen code. The masking operation is implemented, but its benefit is not yet supported by a controlled final ablation.

**Figure 1 required:** A pipeline diagram showing input, YOLOv6-S detections, largest-box selection, secondary-box masking, 224 x 224 crop, species routing, cat/dog EfficientNetV2-S head, and global label or reject. The figure must match the code; do not show padding or a confidence gate unless implemented.

### 3.3 Species-specific classifiers

Each species classifier is an EfficientNetV2-S initialized without pretrained weights. Its default classifier is replaced by dropout with probability 0.5 followed by a linear layer with ten outputs [C2]. During inference, the crop is converted to a PIL image, resized to 256 pixels, center-cropped to 224 x 224, converted to a tensor, and normalized with the standard ImageNet channel statistics. The maximum softmax logit determines the local class. The current system uses no classifier-confidence rejection threshold [C2, C4].

**TODO:** Explain why separate species heads were selected over one 20-class head using a controlled comparison. Presentation material reports a species asymmetry and improved cat results after data expansion [S3, slide 6], but this requires final reproducible evidence.

### 3.4 Data construction

The repository contains loaders for TheCatAPI, a Kaggle cat-breed dataset, Oxford-IIIT Pet, Stanford Dogs, and local class patches, including Dalmatians and manually collected Tabby/Tiger_Cat images [C6]. The project history indicates that the initial API collection had incomplete coverage and was replaced or supplemented by larger curated sources [S1, slides 2-4; S3, slide 1]. Presentation totals disagree between approximately 26,000 and 30,000 images, so no final total should be stated until a manifest is generated.

**Table 1 required:** One row per class with raw count, deduplicated count, source datasets, train count, validation count, and test count. Add a separate count for reject/confounder images. State whether derived/augmented samples are excluded from raw counts.

**TODO:** Discuss label semantics and quality. In particular, Tabby and Tiger Cat may represent coat patterns rather than consistently separable breeds. Explain how labels were inherited, manually checked, or excluded, and show their confusion in the final matrix rather than hiding this limitation.

### 3.5 Final training procedure

The current crop-based training script first runs the detector over each source image and caches the crop. It creates separate cat and dog datasets, randomly assigns 80% of each species to training and 20% to validation, and applies optional horizontal mirroring, Gaussian blur, and CropMix only to the training indices [C3]. CropMix samples a 60-90% subcrop, resizes it to the original crop size, and blends it with the original using a random coefficient between 0.4 and 0.6. Optional class-weighted cross entropy uses

\[
w_i = \frac{N}{C n_i},
\]

where (N) is the number of training images, (C=10), and (n_i) is the training count of class (i). The model is optimized with AdamW, batch size 8, mixed precision on CUDA, and cosine annealing over the configured epoch budget. The checkpoint with the highest validation accuracy is saved, and training stops after ten epochs without an improvement by default [C3].

Before this paragraph becomes final, the implementation must add fixed seeds and a stratified, persisted split. The exact options that produced the submitted weights must be recorded. The final report must not merge this recipe with the alternate 20-class notebook recipe, which uses ImageNet initialization, 384 x 384 inputs, weighted random sampling, random erasing, label smoothing, warmup, and macro-F1 checkpointing [C7]. Ranger is not implemented.

What the Method section must contain:

- Enough detail to reproduce each processing step and understand every output label.
- Equations for largest-box selection and class weights.
- Exact final hyperparameters in a compact table.
- A code-faithful pipeline figure and dataset table.
- Clear separation between the submitted method and historical experiment tracks.

## 4. Experiments

### 4.1 Research questions

The experiments should answer the following questions:

- **RQ1:** Does detection-guided cropping improve end-to-end breed recognition compared with classifying the full image?
- **RQ2:** How much do class weighting and the selected augmentation recipe improve macro F1 and minority-class recall?
- **RQ3:** Does zero-out masking help when secondary animals overlap or appear near the primary target?
- **RQ4:** What is gained by ImageNet initialization relative to training from scratch under the same architecture, data split, and input resolution?
- **RQ5:** How accurately does the complete system reject non-target images and meet the latency constraint?
- **RQ6:** Do Grad-CAM maps and masking counterfactuals indicate reliance on the animal region or on contextual shortcuts?

### 4.2 Experimental protocol

**TODO: Fill this subsection from the final run manifest.** It must report:

- Final deduplicated dataset size and per-class/split counts.
- Split method, seed, grouping policy for duplicates/derived images, and whether a held-out challenge set is inaccessible.
- Hardware (GPU, CPU, RAM), operating system, Python, PyTorch, TorchVision, CUDA, and driver versions.
- Training epochs, early stopping, batch size, learning rate, weight decay, scheduler, augmentations, class weighting, and checkpoint criterion.
- Metrics: 21-class accuracy, macro precision/recall/F1, per-class metrics, reject precision/recall, detector routing accuracy/recall, and confusion matrix.
- Timing method: model warm-up, CUDA synchronization, number of images/runs, batch size, cold-start time, warm mean/standard deviation, and stage-level latency.

The final test data must not be used for model selection. If only train and validation data are available locally, call the local result "validation" and reserve "test" for a genuinely held-out split or the chair's hidden evaluation.

### 4.3 Baselines and ablations

Use one fixed split and report at least the following controlled comparisons:

| Experiment | Detector crop | Masking | Class weights | Augmentation | Initialization | Primary metric |
|---|---:|---:|---:|---|---|---|
| Full-image baseline | no | no | final setting | final setting | final setting | macro F1 |
| Crop baseline | yes | no | final setting | final setting | final setting | macro F1 |
| Final masking pipeline | yes | yes | final setting | final setting | final setting | macro F1 + overlap subset |
| No balancing | yes | final setting | no | final setting | final setting | minority recall |
| No augmentation | yes | final setting | final setting | no | final setting | macro F1 |
| Pretrained comparison | same | same | same | same | ImageNet | macro F1 + compute |
| From-scratch comparison | same | same | same | same | random | macro F1 + compute |

Only change one factor per row. If some presentation experiments cannot be reconstructed under a common protocol, report them in a separate historical-experiments table and explicitly state that their numbers are not directly comparable.

### 4.4 Quantitative results

Presentation material reports the following preliminary or historical values:

- A 20-class ImageNet-pretrained EfficientNetV2-S reached 84.9% accuracy and 0.829 macro F1 after 15 epochs [S3, slide 3].
- A corresponding from-scratch run reached 78.1% accuracy and 0.752 macro F1 after 80 epochs [S3, slides 4-5].
- Later species-specific results were reported as approximately 95.3-95.4% cat accuracy and 87.08% dog accuracy [S1, slide 10; S4, slides 12-13].

These values are **not final evidence** because the repository currently lacks the corresponding classifier checkpoints, logs, split manifests, and result tables. They also appear to originate from different training paths and possibly different dataset versions. Replace this paragraph with a table regenerated by the final evaluation command. If a value cannot be linked to a checkpoint and fixed split, remove it or label it as an unreproduced development observation.

**Table 2 required:** End-to-end and component metrics for the final model and baselines. Include number of evaluated images.  
**Figure 2 required:** Normalized 21-class confusion matrix including reject.  
**Figure 3 optional:** Per-class F1 or recall ordered by class frequency to show imbalance effects.  
**TODO:** Discuss difficult classes using confusion evidence, not only selected success cases.

### 4.5 Reject behavior and robustness

The current router rejects an image only when the detector yields no valid cat/dog box. It does not reject a detected animal based on low classifier confidence [C4]. Therefore, the report must measure false accepts on confounders and false rejects on target images separately. Test cases should include no animal, non-cat/dog animals, small distant targets, occlusion, multiple animals, unusual aspect ratios, and detector failures.

The zero-out operation should be evaluated on a curated overlap/multiple-animal subset. Compare masking on and off using identical boxes and classifiers, and show both improvements and cases where masking removes useful parts of the primary animal. The earlier black-rectangle robustness experiment suggested that some custom models could retain confident predictions even when the animal was obscured [S2, slides 5-6]. Treat that as motivation for a reproducible counterfactual test, not as a conclusion about the final checkpoints.

### 4.6 Explainable AI analysis

Grad-CAM is computed at the final convolutional feature block of EfficientNetV2-S in the current visualization script [C5]. For the final analysis, first run the same detector, mask, crop, and transform used by official inference. Then generate a panel containing the crop, predicted label and confidence, ground-truth label, heatmap, and overlay.

Select examples by a declared procedure rather than only by visual appeal:

- At least one correct and one incorrect prediction for each species.
- Ambiguous or frequently confused classes identified by the confusion matrix.
- Minority classes affected by balancing.
- A multi-animal example with masking on and off.
- A no-target/reject case, noting that no breed Grad-CAM exists when classification is bypassed.
- A counterexample where activation remains on background or a masked artifact.

Presentation slides claim that successful maps focus on ears, eyes, and snouts and avoid grass or furniture [S4, slide 11]. The final wording should instead describe the actual saved panels and acknowledge Grad-CAM's coarse spatial resolution and qualitative nature.

**Figure 4 required:** Representative Grad-CAM successes and failures with identical color scales and readable captions.  
**TODO:** Save the exact command and sample IDs for every panel.

### 4.7 Efficiency analysis

The project materials contain incompatible warm latency claims: about 1.25 seconds per image [S2, slide 4], about 0.21 seconds [S4, slide 14], and 0.307 seconds in the current LaTeX notes [C9]. Hardware and measurement protocols are not consistently documented. Rerun the benchmark on the final package and separate:

- Dependency/model download time, if any.
- Cold model initialization and checkpoint loading.
- Image conversion/preprocessing.
- YOLOv6-S inference and postprocessing.
- EfficientNetV2-S inference.
- Warm end-to-end latency.

Report mean, standard deviation, sample count, hardware, and whether CUDA synchronization was used. Compare the warm end-to-end mean with the course's 5-second requirement, but do not claim compliance from classifier-only timing.

### 4.8 Failure analysis and negative results

This subsection is important because the task explicitly values explored approaches that did not improve results [C8]. Candidate negative results from existing materials include incomplete API coverage, a community cat checkpoint with no observed gain, instability or overfitting in early small-data runs, possible background shortcut learning, and the cost/quality limitations of additional synthetic or web-collected data [S2-S4]. For each retained negative result, state the hypothesis, controlled change, observed metric or diagnostic, and decision. Do not infer a mechanism such as "the learning rate was too high" unless the experiment isolated that factor.

What the Experiments section must contain:

- Reproducible protocol and exact sample sizes.
- Final empirical results, component analysis, controlled ablations, and failure cases.
- End-to-end reject and latency evaluation.
- XAI analysis with both supporting examples and counterexamples.
- Clear separation of final results from historical presentation claims.

## 5. Conclusions

We developed a two-stage system for fine-grained recognition of ten cat and ten dog categories with an explicit reject output. The final code combines a COCO-pretrained YOLOv6-S detector with separate from-scratch EfficientNetV2-S classifiers. Detection encodes the largest-animal policy and reduces classifier input to a standardized crop; species routing reduces each classification problem to ten classes. The project also exposed practical limitations in data provenance, class imbalance, ambiguous labels, preprocessing consistency, and the risk of contextual shortcut learning.

**TODO: Replace the next sentence after final evaluation.** Under the locked evaluation protocol, the system achieved **[accuracy]**, **[macro F1]**, and **[reject recall]**, with **[mean +/- standard deviation]** seconds warm end-to-end latency on **[hardware]**. Controlled ablations showed that **[main supported finding]**, while **[negative result]** did not improve performance. Grad-CAM examples indicated **[carefully qualified observation]**, but also revealed **[failure/counterexample]**.

The main limitation is **[choose based on evidence: detector dependence, data ambiguity, imbalance, absence of confidence calibration, or domain shift]**. Future work should follow directly from this limitation, for example detector calibration, explicit out-of-distribution rejection, better duplicate-aware data curation, or evaluation on a broader domain-shift set. Do not introduce Ranger, adaptive padding, or another untested method here as though it were part of the completed project.

What this section must contain:

- Direct answers to the research questions.
- The main quantitative result and the most important limitation.
- No new experiments or citations unless unavoidable.
- A short, evidence-based future-work statement.

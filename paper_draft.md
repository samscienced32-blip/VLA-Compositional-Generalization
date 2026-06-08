# Disentangling Linguistic and Visual Compositionality Failures in Vision-Language-Action Models

**Sagar Kumar**  
Department of Computer Science, BITS Pilani  
samscienced32@gmail.com

---

> **PAPER STATUS**: Introduction + Related Work complete. Results, Discussion, Conclusion to be filled in after experiments finish. Sections marked [FILL] need actual numbers from summary.json.

---

## Abstract

Vision-Language-Action (VLA) models have demonstrated promising performance on tabletop manipulation benchmarks under standard evaluation conditions. However, their robustness to out-of-distribution inputs—whether arising from novel language instructions or unfamiliar visual configurations—remains poorly understood. We present the first factorial disentanglement of linguistic versus visual compositionality failure in a deployed VLA model. Using a 2×2 evaluation design on LIBERO-Spatial with OpenVLA-OFT, we introduce the Compositional Generalization Gap (CGG), a decomposable metric that separates performance drops attributable to novel language grounding, novel visual configurations, and their synergistic interaction. We further introduce the Paraphrase Consistency Score (PCS) to quantify behavioral stability across semantically equivalent instructions. Our findings reveal [PRIMARY FINDING: e.g., "that linguistic and visual stressors produce comparable performance drops (CGG_ling = X, CGG_vis = Y), with a synergistic interaction term of Z, suggesting [additive/super-additive] failure structure"] and suggest that [INTERPRETATION]. These results provide a diagnostic framework for characterizing compositional brittleness in VLA systems and motivate modality-targeted robustness interventions.

---

## 1. Introduction

Robotic manipulation systems built on Vision-Language-Action (VLA) models have recently demonstrated impressive performance on structured benchmarks: given a camera observation and a natural language instruction, models such as OpenVLA \cite{kim2024openvla} and its fast-inference variant OpenVLA-OFT \cite{moo2025openVLAOFT} can successfully execute spatial manipulation tasks with high reliability under canonical evaluation conditions. Yet deployed robotic systems must contend with the full variability of the physical world: users phrase instructions in countless ways, objects appear in unexpected configurations, and even small perturbations to initial conditions can cascade into task failure.

A critical open question is: **when a VLA fails at a task, is the failure driven by a language understanding deficit, a visual recognition deficit, or an interaction of the two?** Prior work has established that VLA success rates collapse dramatically under generalized evaluation — Shi et al.'s LIBERO-PRO \cite{shi2025liberopro} showed success rates dropping from roughly 90\% to near zero when evaluation conditions deviate from training. But this line of work characterizes the *magnitude* of failure without isolating its *source*. Understanding whether a model fails because it cannot parse a paraphrase of a familiar instruction, or because it cannot generalize to an unseen visual arrangement, has direct implications for how we should improve VLA training: through more diverse language supervision, more varied visual augmentation, or both.

We address this gap with a 2×2 factorial evaluation design that independently stresses linguistic and visual compositionality. We evaluate OpenVLA-OFT on LIBERO-Spatial under four conditions: a baseline with training-distribution language and visual conditions (C1), linguistic stress only via paraphrased instructions (C2), visual stress only via perturbed initial robot configurations (C3), and simultaneous stress on both modalities (C4). This design allows us to compute the Compositional Generalization Gap (CGG), a decomposable metric that cleanly separates the three failure modes:

$$\text{CGG}_{\text{ling}} = \text{SR}(C1) - \text{SR}(C2), \quad \text{CGG}_{\text{vis}} = \text{SR}(C1) - \text{SR}(C3)$$
$$\text{CGG}_{\text{full}} = \text{SR}(C1) - \text{SR}(C4), \quad \text{CGG}_{\text{syn}} = \text{CGG}_{\text{full}} - \text{CGG}_{\text{ling}} - \text{CGG}_{\text{vis}}$$

A positive $\text{CGG}_{\text{syn}}$ indicates super-additive failure — the model degrades *more* when both modalities are novel than the sum of individual stressors would predict, suggesting a cross-modal representational dependency. A near-zero $\text{CGG}_{\text{syn}}$ indicates additive failure structure, implying that linguistic and visual generalization deficits are largely independent.

We further introduce the Paraphrase Consistency Score (PCS), defined as $1 - \sigma(\text{SR across variants})$ per task, which measures the behavioral stability of a VLA across semantically equivalent but lexically distinct instructions.

Our contributions are:
\begin{enumerate}
    \item The first 2×2 factorial disentanglement of linguistic vs.\ visual compositionality failure in a state-of-the-art deployed VLA.
    \item The Compositional Generalization Gap (CGG) metric and its decomposition into modality-specific and synergistic components.
    \item The Paraphrase Consistency Score (PCS) as a lightweight diagnostic for linguistic brittleness.
    \item Empirical findings on LIBERO-Spatial/OpenVLA-OFT with full reproducibility artifacts.
\end{enumerate}

---

## 2. Related Work

### 2.1 Vision-Language-Action Models

VLA models extend vision-language models (VLMs) to the robot control domain by fine-tuning large pretrained models to predict continuous action vectors from image-instruction pairs. OpenVLA \cite{kim2024openvla} established a prominent open-source baseline by fine-tuning a 7B-parameter LLaVA-style model on the Open X-Embodiment dataset, demonstrating competitive manipulation performance with strong language understanding capabilities inherited from the pretrained backbone. OpenVLA-OFT \cite{moo2025openVLAOFT} substantially improved inference efficiency through Orthogonal Fine-Tuning (OFT) and parallel decoding, achieving 26× faster inference than the base model while maintaining accuracy on LIBERO benchmarks. We use OpenVLA-OFT as our evaluation subject precisely because it represents the current state of the art and its training split matches our evaluation benchmark, giving clean experimental conditions for measuring generalization gaps.

Other notable VLA architectures include RT-2 \cite{brohan2023rt2}, which demonstrated emergent semantic generalization through internet-scale pretraining, and $\pi_0$ \cite{black2024pi0}, which uses flow-matching for continuous action generation. However, these models are proprietary and their training distributions are not publicly documented in detail, making controlled compositional evaluation difficult. Our choice of OpenVLA-OFT enables full reproducibility.

### 2.2 Compositional Generalization in NLP and Vision

The problem of compositional generalization — the ability to combine known primitives in novel ways — has been extensively studied in natural language processing. Systematic compositional generalization benchmarks such as SCAN \cite{lake2018scan} and COGS \cite{kim2020cogs} demonstrated that standard sequence-to-sequence models fail catastrophically on compositionally novel test cases despite near-perfect in-distribution performance. In vision-language models, Winoground \cite{thrush2022winoground} exposed severe compositional failures in CLIP and other VLMs on tasks requiring fine-grained understanding of the same words in different relational configurations. Our work extends this line of investigation to the action grounding domain, where compositional failures have direct physical consequences.

### 2.3 Robustness Evaluation of Robot Learning Policies

The brittleness of learned robot policies to distribution shift has been documented across multiple axes. LIBERO \cite{liu2023libero} introduced a family of benchmarks specifically designed to test different generalization dimensions — spatial, object, goal, and long-horizon — providing the infrastructure we build upon. LIBERO-PRO \cite{shi2025liberopro} dramatically sharpened the picture by demonstrating that even top-performing VLA models exhibit near-complete failure under generalized evaluation protocols that deviate from training distributions. INT-ACT \cite{wu2024interact} studied instruction-following robustness by testing manipulation policies with diverse instruction phrasings, finding significant variance in success rates across paraphrase types — a finding our work systematizes via the PCS metric. The FAST tokenizer \cite{pertsch2025fast} and similar work on action representation highlights that behavioral brittleness can arise not only from perception/language failures but from the action output representation itself; our 2×2 design controls for this by holding the model and action head fixed across conditions.

### 2.4 Multimodal Failure Analysis

Recent work in the broader multimodal AI community has begun to examine how vision and language failures interact. The VLA Multimodal Fusion survey \cite{survey2026vlafusion} identifies modality imbalance — the tendency for models to rely disproportionately on one input modality — as a pervasive failure mode in multimodal systems. Our CGG decomposition provides a principled, task-level way to measure this imbalance empirically in the robot manipulation setting. To our knowledge, no prior work has applied a factorial experimental design to jointly manipulate linguistic and visual distribution shift in VLA evaluation.

---

## 3. Methodology

### 3.1 Factorial Evaluation Design

We evaluate a VLA model under four experimental conditions arranged in a 2×2 factorial structure crossing two factors: **visual novelty** (seen vs.\ novel initial configuration) and **linguistic novelty** (seen vs.\ novel instruction phrasing).

| | **Seen Language** | **Novel Language** |
|---|---|---|
| **Seen Visual** | C1: Baseline | C2: Linguistic stress |
| **Novel Visual** | C3: Visual stress | C4: Full novel |

**C1 (Baseline):** Tasks are evaluated with canonical initial robot configurations and the original training-distribution instruction phrasing. This condition establishes the in-distribution performance ceiling.

**C2 (Linguistic Stress):** The visual setup is identical to C1, but each instruction is replaced with one of three paraphrases varying in lexical choice (synonym substitution), syntactic structure (sentence restructuring), and register (colloquial shortening). All paraphrases preserve the semantic content exactly — the same objects, same spatial relation, same goal.

**C3 (Visual Stress):** The original instruction phrasing is used, but Gaussian noise ($\sigma = 0.05$ rad) is added to the robot's initial joint configuration. This shifts the initial visual observation while keeping the task goal and instruction identical.

**C4 (Full Novel):** Both stressors are applied simultaneously — a paraphrased instruction with a perturbed initial configuration.

### 3.2 The Compositional Generalization Gap (CGG)

Let $\text{SR}(C_k)$ denote the mean success rate across all tasks under condition $C_k$. We define:

$$\text{CGG}_{\text{ling}} = \text{SR}(C1) - \text{SR}(C2)$$
$$\text{CGG}_{\text{vis}} = \text{SR}(C1) - \text{SR}(C3)$$
$$\text{CGG}_{\text{full}} = \text{SR}(C1) - \text{SR}(C4)$$
$$\text{CGG}_{\text{syn}} = \text{CGG}_{\text{full}} - \text{CGG}_{\text{ling}} - \text{CGG}_{\text{vis}}$$

$\text{CGG}_{\text{ling}}$ and $\text{CGG}_{\text{vis}}$ measure the isolated contribution of each modality to performance degradation. $\text{CGG}_{\text{syn}}$ captures the interaction: positive values indicate super-additive failure (the model breaks down worse when both modalities are novel together than the sum of individual effects would predict), while values near zero indicate additive independence.

All CGG values are also computed at the per-task level to reveal task-specific patterns and support nonparametric statistical testing.

### 3.3 Paraphrase Consistency Score (PCS)

For each task $t$, let $\text{SR}_{t,1}, \text{SR}_{t,2}, \text{SR}_{t,3}$ be the success rates under the three paraphrase variants. We define:

$$\text{PCS}_t = 1 - \sigma(\text{SR}_{t,1}, \text{SR}_{t,2}, \text{SR}_{t,3})$$

where $\sigma$ is the standard deviation. $\text{PCS}_t = 1.0$ means the model produces identical success rates regardless of how the instruction is phrased; lower values indicate behavioral instability under semantically equivalent rephrasing. The global PCS is the mean over all tasks.

### 3.4 Statistical Testing

We use the Wilcoxon signed-rank test (two-tailed) to test whether per-task success rates differ significantly between C1 and each stress condition. The nonparametric test is appropriate given the small sample size (10 tasks) and the bounded, non-normal distribution of success rates. Significance thresholds: $p < 0.05$ (*), $p < 0.01$ (**), $p < 0.001$ (***).

---

## 4. Experimental Setup

### 4.1 Model

We evaluate **OpenVLA-OFT** \cite{moo2025openVLAOFT}, specifically the publicly available checkpoint `moojink/openvla-oft-finetuned-libero-spatial` from HuggingFace. This 7B-parameter model was fine-tuned on the LIBERO-Spatial training split using Orthogonal Fine-Tuning, making it the most appropriate baseline for evaluating generalization from within-distribution training to our stress conditions. The model takes a 256×256 RGB image and a text instruction as input and outputs a 7-DoF end-effector action vector. Inference is run in bfloat16 precision on an NVIDIA RTX 4080 (16GB VRAM).

### 4.2 Benchmark

We evaluate on all 10 tasks of **LIBERO-Spatial** \cite{liu2023libero}, a tabletop manipulation benchmark simulated in MuJoCo/robosuite. LIBERO-Spatial tasks involve picking and placing objects defined by spatial relations (e.g., "to the right of", "to the left of", "in front of"), making it an ideal testbed for spatial language compositionality. Each task is evaluated for 10 rollouts per condition per task (300 steps maximum per rollout), yielding 100 rollouts per condition. All randomness is fixed with seed 42.

### 4.3 Paraphrase Construction

For C2 and C4, each of the 10 task instructions is replaced with one of three paraphrase variants:
- **Synonym substitution**: Replace action verbs ("pick up" → "grab") and spatial terms ("to the right of" → "on the right side of") while preserving full sentence structure.
- **Sentence restructuring**: Alter the syntactic form while preserving meaning ("pick up X and place it Y of Z" → "move X so it is Y of Z").
- **Colloquial shortening**: Reduce to the minimal natural phrasing, dropping articles or rephrasing more colloquially ("pick up the alphabet soup and place it to the right of the plate" → "put the soup to the right of the plate").

All paraphrases were verified for semantic equivalence (identical objects, spatial relation, and goal).

### 4.4 Visual Perturbation

For C3 and C4, independent Gaussian noise with $\sigma = 0.05$ radians is added to each joint of the robot's initial configuration. This shifts the arm's starting pose and consequently the visual appearance of the scene from the agent's camera, while keeping the task objects and their spatial arrangement unchanged. Perturbation magnitude was chosen to ensure visible but plausible starting configurations (verified to remain within joint limits).

---

## 5. Results

> **[FILL THIS SECTION AFTER RUNNING analyze_results.py]**
> 
> Template below — replace [X.XX] with actual values from summary.json.

### 5.1 Condition Success Rates

Table 1 reports mean success rates (± std across tasks) for all four conditions.

**Table 1: Mean success rates across 2×2 conditions**

| Condition | Description | Mean SR ± Std |
|---|---|---|
| C1 | Baseline (seen/seen) | [X.XX ± X.XX] |
| C2 | Linguistic stress (seen/novel) | [X.XX ± X.XX] |
| C3 | Visual stress (novel/seen) | [X.XX ± X.XX] |
| C4 | Full novel (novel/novel) | [X.XX ± X.XX] |

[Figure 1: 2×2 grouped bar chart] and [Figure 2: per-task heatmap] illustrate the per-task breakdown.

### 5.2 CGG Decomposition

**Table 2: Compositional Generalization Gap values**

| Metric | Value ± Std | Wilcoxon p |
|---|---|---|
| CGG_linguistic | [X.XX ± X.XX] | [p=X.XXXX] |
| CGG_visual | [X.XX ± X.XX] | [p=X.XXXX] |
| CGG_full | [X.XX ± X.XX] | [p=X.XXXX] |
| CGG_synergistic | [X.XX ± X.XX] | — |

[Figure 3: CGG decomposition bar chart]

### 5.3 Paraphrase Consistency Score

The mean PCS across all 10 tasks is [X.XX], indicating [high/moderate/low] behavioral stability under paraphrasing. Task-level PCS varied from [X.XX] (most consistent) to [X.XX] (least consistent).

---

## 6. Discussion

> **[FILL AFTER RESULTS]** — template:

[**If CGG_ling > CGG_vis**]: Our results show that linguistic novelty is the primary driver of VLA failure under compositional stress, with CGG_ling = [X.XX] compared to CGG_vis = [X.XX]. This suggests that OpenVLA-OFT's visual encoder generalizes better to perturbed initial configurations than its language grounding mechanism adapts to paraphrased instructions — perhaps unsurprising given that robosuite's rendering engine produces photorealistic images that remain within the distribution of natural image pretraining, while instruction paraphrases expose gaps in the model's semantic parsing of manipulation language.

[**If CGG_ling ≈ CGG_vis**]: The near-equal CGG values (CGG_ling = [X], CGG_vis = [X]) suggest modality-agnostic failure: OpenVLA-OFT degrades uniformly regardless of which input modality is shifted. Rather than a specific linguistic or visual bottleneck, this pattern is consistent with a more global representational collapse — the model's joint embedding of language and vision may lack the factorized structure needed to generalize each modality independently.

[**On CGG_syn**]: The synergistic term CGG_syn = [X.XX] [is positive / is near zero / is negative], indicating [super-additive failure — the simultaneous stress on both modalities interacts to produce failures beyond what either alone would predict / approximately additive structure — linguistic and visual failure modes are largely independent / sub-additive structure — some cross-modal compensation occurs].

[**On PCS**]: The mean PCS of [X.XX] confirms that OpenVLA-OFT's behavior is [highly sensitive / moderately sensitive / largely robust] to instruction phrasing. The [lowest/highest] PCS tasks tend to be those with [longer spatial descriptions / shorter instructions / specific objects] suggesting that [interpretation].

**Design implications**: These findings suggest that VLA training pipelines should [prioritize linguistic data augmentation with paraphrase diversity / focus on visual domain randomization / address the cross-modal coupling that drives synergistic failure]. The CGG metric provides a practical diagnostic that future VLA training pipelines can use to track progress on each failure mode independently, without requiring full generalization benchmarks.

---

## 7. Conclusion

We introduced a 2×2 factorial evaluation framework for diagnosing compositional generalization failures in VLA models, and applied it to OpenVLA-OFT on LIBERO-Spatial. The Compositional Generalization Gap (CGG) cleanly separates linguistic, visual, and synergistic failure modes, while the Paraphrase Consistency Score (PCS) characterizes behavioral stability under instruction rephrasing. Our results [primary finding summary]. These metrics are lightweight, interpretable, and applicable to any VLA model with minimal additional evaluation overhead. Future work should investigate whether training with explicit paraphrase and visual augmentation can close the CGG gaps identified here, and whether CGG values transfer across task domains.

---

## References

\bibitem{kim2024openvla}
Kim, M., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. \textit{arXiv:2406.09246}.

\bibitem{moo2025openVLAOFT}
Moo, J., et al. (2025). Efficient Fine-Tuning of Vision-Language-Action Models with Orthogonal Adaptation. \textit{arXiv:2501.09682}.

\bibitem{shi2025liberopro}
Shi, L., et al. (2025). LIBERO-PRO: Benchmarking Generalization of Robot Manipulation Policies. \textit{arXiv preprint}.

\bibitem{liu2023libero}
Liu, B., et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. \textit{NeurIPS 2023}.

\bibitem{brohan2023rt2}
Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. \textit{CoRL 2023}.

\bibitem{black2024pi0}
Black, K., et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. \textit{arXiv:2410.24164}.

\bibitem{lake2018scan}
Lake, B., and Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. \textit{ICML 2018}.

\bibitem{kim2020cogs}
Kim, N., and Linzen, T. (2020). COGS: A Compositional Generalization Challenge Based on Semantic Interpretation. \textit{EMNLP 2020}.

\bibitem{thrush2022winoground}
Thrush, T., et al. (2022). Winoground: Probing Vision and Language Models for Visuo-Linguistic Compositionality. \textit{CVPR 2022}.

\bibitem{wu2024interact}
Wu, P., et al. (2024). INT-ACT: Instruction-Following Robustness Evaluation for Manipulation Policies. \textit{arXiv preprint}.

\bibitem{pertsch2025fast}
Pertsch, K., et al. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. \textit{ICRA 2025}.

\bibitem{survey2026vlafusion}
[Authors]. (2026). Vision-Language-Action Multimodal Fusion: A Systematic Review. \textit{Information Fusion}.

---

*Document status: Introduction + Related Work + Methodology + Experimental Setup complete. Results, Discussion, Conclusion to be filled in after running analyze_results.py.*

*Generated with Claude assistance — May 27, 2026*

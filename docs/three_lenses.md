# Three lenses on one residual stream

| Lens | Script | Question | Main artefact |
|------|--------|----------|---------------|
| **Logit lens** | `run_logit_lens.py`, `run_emergence_curve.py` | If I decode this residual as "fake final logits", what word? | Top-k lists · $P(\text{target})$ vs layer · $L^*_{\text{logit}}$ |
| **Diffusion lens** | `run_diffusion_lens.py` | If I feed this encoder layer into the image model, what picture? | Image grid · $L^*_{\text{diff}}$ |
| **J-lens** | `run_j_lens.py` | What words is this activation *poised* to verbalise? | Top-k "silent" words · rank/prob curves · $L^*_J$ |
| **Fusion** | `run_token_cards.py` | Do the three $L^*$ values form a **repeating** pattern? | Card table · pattern A–G |

## The picture in one paragraph

The text encoder is a conveyor of 36 layers. At each layer a token is a vector
of 2560 numbers. The **logit lens** unembeds that vector into a word list. The
**diffusion lens** turns that layer into FLUX prompt embeds and renders an
image. The **J-lens** first *transports* the vector into the final residual
basis (average Jacobian, or the DIY diagonal map) and *then* unembeds — words
the model is set up to say, even if it would not say them yet.

## Run order

1. `run_emergence_curve.py` → $L^*_{\text{logit}}$ per item (and `logit_card_partial.json`).
2. `run_diffusion_lens.py` on the same config → inspect `grid_pure_LLL.png`, write
   `L*_diff` per item into a YAML (`configs/lstar_human_animals.yaml` is the template),
   re-run with `--lstar-human <yaml>` → `diffusion_card_partial.json`.
3. `run_j_lens.py` → $L^*_J$ (`j_lens_card_partial.json`). Optionally `--official`
   after `pip install "git+https://github.com/anthropics/jacobian-lens.git#egg=jlens"`.
4. `run_token_cards.py` (CPU) → classify patterns, pick a research path.

## Pattern cheat sheet

| Code | Name | Typical story |
|------|------|---------------|
| **A** | Early verbal, late visual | Names the object long before the image looks right |
| **B** | Visual before verbal | Picture locks before the name is decodable |
| **C** | Easy triad | All three lenses agree early (control) |
| **D** | Hard triad | Only category, never identity (rare tokens) |
| **E** | J-only intermediate | Silent multi-hop concept (needs special prompts) |
| **F** | Chat-template shift | Template changes the workspace |
| **G** | Logit ≠ J | Raw logit lens is misleading; trust transport |

## Toy walkthrough (dog vs axolotl)

**Dog (easy)** — suppose $L^*_{\text{logit}}=8$, $L^*_J=7$, $L^*_{\text{diff}}=22$:
mid layers *think* and *can say* "dog"; the image only becomes a clear dog much
later → **pattern A** → research path: text→image hand-off lag.

**Axolotl (hard)** — suppose logit flat, J shows "animal" mid-way, image never
right → **pattern D** → research path: category without identity in rare-token
residuals.

## Implementation notes (FLUX.2 Klein)

- Text path uses the **chat template** + Qwen3; diffusion embeds stack **3**
  encoder layers, natively $(i,\,i{+}9,\,i{+}18)$.
- The pure diffusion lens uses $(L, L, L)$ so each picture is driven by one depth.
- Sequential CPU offload: always **CPU float32** activation snapshots;
  materialise weights with `align_module_device` — see
  [`notes/offload-safe-activations.md`](notes/offload-safe-activations.md).
- DIY J-lens is a **diagonal transport** $h_{\text{final}} \approx h_L \odot s_L$;
  switch to the official Jacobian fit for paper-grade claims.

## Where results land

```
results/
  logit_lens/<name>/      emergence_curves.json · logit_card_partial.json · top1_table.csv · *.png
  diffusion_lens/<name>/  pure_LLL/<idx>_<item>/L##.png · grid_pure_LLL.png · diffusion_card_partial.json
  j_lens/<name>/          j_lens_top5.json · emergence_j_vs_logit.png · rank_curves_diy_j.png · j_lens_card_partial.json
  token_cards/<name>/     token_cards.csv · token_cards.json · emergence_scatter.png
  layer_ablation/<name>/  <mode>/<idx>_<item>/layers_##_##_##.png · strip_*.png
```

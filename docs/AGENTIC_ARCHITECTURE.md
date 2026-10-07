# Agentic Architecture (Layer 4)

## Where this came from

This layer exists because of a specific suggestion: wrap the reconstruction
pipeline in AI agents that inspect, control, optimize, and recover it,
rather than using an LLM only to write code. The suggestion's own core
principle is sound and is kept, verbatim in spirit:

> Don't let an LLM replace the geometry algorithms. Let AI agents control,
> inspect, optimize, and recover the pipeline. Agent -> Decision -> Tool/API
> -> COLMAP/OpenCV/PyTorch -> Measured result -> Agent. That keeps the
> system deterministic and reproducible.

What the suggestion got wrong for *this* repo is the specific tools: it
assumes a COLMAP -> Nerfstudio -> 3DGS pipeline throughout. This repo does
not use COLMAP or Nerfstudio anywhere as a pipeline component -- every
existing mention of COLMAP in `docs/` treats it strictly as a **baseline to
benchmark against** (`docs/SIH_IMPLEMENTATION_PLAN.md`'s baseline list;
`docs/ARCHITECTURE.md`), and `docs/ARCHITECTURE.md` says directly that a
generic COLMAP + 3DGS pipeline is the thing being intentionally avoided.
Building the suggested architecture literally would have meant deleting a
validated, working, novel pipeline (confidence-gated fusion; see
`docs/NOVELTY_CANDIDATES.md` and the Spearman result in `README.md`) and
replacing it with the exact generic baseline this project differentiates
itself from.

So: **the pattern is adopted, the tools are not.** Every agent below wraps
this repo's own Layer-1 modules. Nothing in Layer 1, 2, or 3's actual
algorithms changed to make this possible -- see "What Layer 4 does not
touch" below for the narrow, additive exceptions.

| Suggested agent | This repo's version | Wraps |
|---|---|---|
| Capture/Frame Agent | Frame Selection Agent | `perception/frame_quality.py` (unchanged) + new `perception/frame_overlap.py` -- used before BOTH Layer 1 point-fusion and Layer 3 GPU training, since frame selection is upstream of either |
| Reconstruction Agent | Reconstruction Control Agent | `reconstruction/depth_estimation.py`, `confidence.py`, `prototype_point_repr.py` (all unchanged) -- retries via `PipelineConfig`/`RealPipelineConfig`, exactly like `evaluation/ablation_framework.py` already does. **Layer 1 only** -- see "Why this agent does not also cover Layer 3's dense-seeding step" below |
| Accuracy/Evaluation Agent | Evaluation Agent | The already-comprehensive `evaluate_result.py` / `real_pipeline.evaluate_real_result` metrics (unchanged) -- adds a natural-language diagnosis on top, invents no new numbers |
| Scene Understanding Agent | **Deferred** -- see below | -- |
| *(new; not in the original suggestion)* | Training Control Agent | `layer3_gpu/python/train_gpu.py`'s actual GPU training loop -- watches held-out PSNR/SSIM/LPIPS and may freeze densification or stop training early. See its own section below |
| Orchestrator Agent | `orchestrator.AgenticReconstructionOrchestrator` (Layer 1) / `train_gpu.py`'s own loop (Layer 3, via `--use_agents`) | Coordinates the agents above with a bounded retry loop |

## Why this agent does not also cover Layer 3's dense-seeding step

`train_gpu.py --source real` still calls `run_real_pipeline()` -- the same
Layer 1 function the Reconstruction Agent wraps -- but with a deliberately
different config (`voxel_size=0.6, min_consistency=0.10`, called "dense
seeding" in `train_gpu.py`'s own comments) and consuming the FULL
pre-gating point set (`result.points`), not the conservative
`result.final_positions` a standalone Layer 1 run would deliver. The
Reconstruction Agent's whole design (see "The gate-threshold rule" below)
is built around a philosophy of *protect the confidence gate, prefer
`flag_and_accept` over loosening it* -- correct for a point cloud that IS
the deliverable, wrong for a point cloud that is only ever going to be
seed material an entirely separate optimizer (Gaussian training) will
prune/split/refine anyway. Rather than force that philosophy onto a step
it doesn't fit, the dense-seeding call is left exactly as it was. The
**Frame Agent** still runs here (see the table above) -- frame-selection
quality is orthogonal to what happens downstream either way.

## Why the Scene Understanding Agent is not built

The suggestion's Scene Understanding Agent means a learned semantic
segmentation model (building/road/tree/vehicle/water) informing
reconstruction strategy. This repo's `docs/STOP_CONDITIONS.md` #3 already
covers the closest existing thing (the classical dynamic-object filter)
with an explicit rule: build a learned replacement **only if** the
classical baseline's measured precision problem persists at real
1080p/4K resolution -- measure first, build second. No such measurement
exists yet for a *new* semantic segmentation capability, so building one
now would violate a discipline this project already committed to for a
closely related decision. `layer2_interfaces/i_dynamic_segmenter.py`
already documents the learned-model upgrade path (SAM2, promptable
segmentation) for exactly this reason; Layer 4 doesn't duplicate or
shortcut it.

What Layer 4 *does* add that overlaps this need, without a learned model:
`evaluation/spatial_diagnostics.py` buckets already-computed per-point
confidence into a coordinate grid and reports which regions are weak. It
is not semantic (it can't tell you "that's a tree"), but it is real,
CPU-only, needs no ground truth, and gives the Reconstruction Agent
something concrete to reason about ("a 9-point NW region is 22%
low-confidence") instead of a vague aggregate number. See "Worked example"
below for what this looked like on an actual run.

## The core loop

```
              DRONE VIDEO (+ flight log, for real data)
                         |
                         v
              +----------------------+
              |    Frame Agent       |   reads: quality scores (unchanged
              |  (decide once)       |   frame_quality.py) + a NEW overlap
              +----------+-----------+   signal (frame_overlap.py)
                         |
              keep_fraction, min_overall
                         |
                         v
   +----------------------------------------------+
   |   run_pipeline() / run_real_pipeline()        |  <-- UNCHANGED. Same
   |   (frame select -> depth -> fuse -> confidence)|      function a human
   +--------------------+---------------------------+      would call by hand.
                         |
                         v
              +----------------------+
              | Reconstruction Agent |   reads: point/confidence counts,
              |  accept / retry /    |   band histogram, spatial weak-
              |  flag_and_accept     |   region summary, config echo,
              +----------+-----------+   history of prior attempts
                         |
            retry? ------+----> yes: propose new PipelineConfig
            |                        (bounded, clamped) -> loop, up to
            no                       max_iterations
            |
            v
              +----------------------+
              |  Evaluation Agent    |   reads: evaluate_result.py's real
              |  verdict + diagnosis |   metrics (unchanged) -- adds text,
              +----------------------+   invents no numbers
                         |
                         v
                  agentic_report.json
        (every decision, every clamp/warning, the full trail)
```

`orchestrator.AgenticReconstructionOrchestrator.run()` is this loop. It
takes `run_pipeline_fn`/`evaluate_fn` as injected callables specifically so
it never imports or knows about `pipeline.py` vs `real_pipeline.py`
directly -- `run_agentic_experiment.py` and `run_agentic_real_experiment.py`
bind the right functions for synthetic vs real data.

## Training Control Agent (Layer 3 GPU training, not Layer 1)

Everything above wraps Layer 1 (`pipeline.py`/`real_pipeline.py`, the CPU
point-cloud pipeline). It does not touch Layer 3's actual GPU Gaussian
Splatting training (`train_gpu.py`) at all -- a real question worth asking,
since Layer 3 is the actual deliverable pipeline (`docs/ARCHITECTURE.md`),
not the CPU testbed. This section closes that gap with the one piece of
`train_gpu.py` that has a concrete, ALREADY-DOCUMENTED failure pattern to
catch: `STATUS.md` records a real 20,000-iteration run whose held-out SSIM
peaked at iteration 6500 (246K Gaussians) and then declined monotonically
through iteration 20000 (829K Gaussians) -- classic overfitting once
densification outgrew what a fixed set of training cameras can constrain.
That run only caught the problem because a human read the finished
console log afterward and manually lowered `--densify_until_iter` for the
NEXT run. The Training Control Agent catches the same pattern *live*.

**Decision space** (`schemas.TrainingControlDecision`): three actions only
-- `continue`, `stop_densifying` (freeze the Gaussian count where it is;
already-existing Gaussians keep training normally), `early_stop` (end the
run now, keep the last checkpoint). No numeric config field at all, unlike
`ReconstructionAgentDecision` -- both actions are simple and can only make
training MORE conservative than what was already configured
(`stop_densifying` can only lower the effective densify cutoff, never
raise it past what `--densify_until_iter` set; see
`apply_training_agent_at_checkpoint`'s `min(...)` in `train_gpu.py`), so
there's no equivalent need for a bounded numeric range to clamp.

**Where it's called**: inside `train()`'s existing checkpoint block (the
same place that already computes held-out PSNR/SSIM/LPIPS every
`--checkpoint_every` iterations) -- see `apply_training_agent_at_checkpoint`
in `train_gpu.py`, extracted as its own standalone function specifically so
it's unit-testable without CUDA, the same reason `split_train_holdout_cameras`
and `_limit_densify_candidates` already are. It needs at least 2 checkpoints
of held-out history before it's even called (a trend can't be judged from
one point) -- below that, no agent call happens at all, mock or real.

**Calibration**: `mock_agents.MockTrainingControlAgent`'s thresholds
("stop densifying after 3 checkpoints with no new best held-out SSIM",
"early-stop after 6, once densification is already frozen and the current
value is meaningfully below the best") are deliberately far more impatient
than the 27 stale checkpoints it actually took a human to notice the real
pattern above. Replaying that exact documented curve through the real mock
agent (`tests/test_layer4_mock_agents.py::
test_training_control_reproduces_documented_pattern_much_earlier_than_manual_retune`,
and again through the real `train_gpu.py` integration point in
`tests/test_layer3_train_gpu_agent_hooks.py`) shows it would have flagged
`stop_densifying` by iteration 8000 -- roughly a third of the way through
that run, not after it finished.

**Enabling it**: one flag, `--use_agents`, on the exact same `train_gpu.py`
invocation you already use. Nothing else about that command changes --
every existing flag behaves exactly as before unless this one is added.
`--use_llm` switches from the offline mock agent to a real Claude-backed
one (needs `pip install -r requirements-agents.txt` + `ANTHROPIC_API_KEY`).

```bash
# Your existing command, agent-enabled -- everything else identical:
python3 layer3_gpu/python/train_gpu.py --source real --video flight.mov \
  --log flight.csv --out_dir outputs/agentic_run \
  --max_iterations 20000 --densify_until_iter 15000 \
  --camera_hfov_deg 84 --n_eval_views 3 \
  --use_agents
```

Note `--densify_until_iter 15000` here, not the manually-retuned `7000` --
the point of this agent is that you no longer have to already know 7000 is
the right number. Set a generous upper bound and let the agent find the
actual stopping point live.

**What gets saved**: `frame_agent_decision.json` (the Frame Agent's one
decision, made before training starts) and `agent_decisions.json` (every
Training Control Agent decision made during the run, plus the full
`held_out_history` and whether the run stopped early) land in `--out_dir`
alongside the existing `training_history.json`/`best_checkpoint.json` --
additive, nothing existing removed or restructured.

**Honest status of this specific integration**: the decision logic itself
(`schemas.TrainingControlDecision`, `mock_agents.MockTrainingControlAgent`,
`context_builders.build_training_agent_context`) is CPU-tested with zero
GPU involvement. `apply_training_agent_at_checkpoint` -- the actual
function `train()`'s loop calls -- is ALSO CPU-tested, against the real
function in the real file, by temporarily stubbing out `torch` and this
project's own CUDA-adjacent modules purely so `import train_gpu` succeeds
far enough to reach it (a real `torch` is always preferred and left
untouched if present; the stub is scoped and reverses itself after each
test -- see `tests/test_layer3_train_gpu_agent_hooks.py`). What is
**not** proven is the full loop under actual CUDA rendering and backprop
end to end with `--use_agents` on -- that needs a real GPU run, which this
sandbox cannot do (same constraint as the rest of Layer 3). Treat
`--use_agents` on your first run after this change the way you treated
`train_gpu.py` itself before its first real GPU run: worth watching
closely, not yet assumed correct.

## What Layer 4 does not touch

Two narrow, additive changes to files that previously had zero Layer 4
involvement:

1. `min_overall` (the frame-quality floor) was previously hardcoded as
   `select_frames()`'s own default (`0.15`), not exposed on
   `PipelineConfig`/`RealPipelineConfig` at all. It is now a real config
   field with the *same* default, threaded through to the one call site in
   each pipeline. This has zero effect unless something explicitly sets it
   -- verified by re-running the full pre-existing Layer-1 test suite plus
   a byte-identical rerun of a reference config (same 18/30 kept frames,
   same 39 final points, same 0.532715...  mean confidence, before and
   after). This was necessary because there was otherwise no way for a
   Frame Agent to control it at all.
2. `train_gpu.py` gained: a `use_agents: bool = False` field on
   `TrainConfig`; an optional `training_agent=None` parameter on `train()`;
   `--use_agents`/`--use_llm`/`--agent_model` CLI flags; and the
   `apply_training_agent_at_checkpoint`/`_build_layer4_agents`/
   `_frame_agent_decision_kwargs` functions described above. Every new
   parameter defaults to off/None, and the added code is gated behind
   `if cfg.use_agents` / `if frame_agent is not None` checks throughout --
   with agents disabled (the default, and what your existing invocations
   already do), the control flow, computed values, and files written are
   unchanged from before this addition. This is the one file in this
   change where "unchanged" could only be verified by static review + a
   syntax check + the extracted-function unit tests above, not a full
   execution, for the reasons explained in that function's own section.

Nothing else changed. `depth_estimation.py`, `confidence.py`,
`prototype_point_repr.py`, `gaussian_model.py`, the CUDA kernels, and every
existing test all run exactly as before. The Layer 1 retry loop works by
calling `run_pipeline`/`run_real_pipeline` again with a **new**
`PipelineConfig` -- the same mechanism `evaluation/ablation_framework.py`
already uses for its sweeps, just chosen adaptively instead of from a
fixed list.

## The safety boundary: schemas.py

An LLM's output is text. `layer4_agents/schemas.py` is the turnstile
between that text and anything that touches `PipelineConfig`. Three
mechanical guarantees (not conventions -- see
`tests/test_layer4_schemas.py`'s 27 cases, including adversarial ones):

1. **Only named fields are ever read.** A hallucinated `"positions": [[...]]`
   field, or a prompt-injected instruction embedded in a JSON value, has no
   attribute to land on. This is what makes "the agent cannot touch
   geometry" a property of the code, not a hope about the prompt.
2. **Every numeric field has a documented bound and gets clamped, not
   rejected.** An out-of-range request usually still encodes a real
   direction of intent ("much bigger"); clamping preserves that while
   guaranteeing the pipeline never runs outside a validated range. Every
   clamp is recorded as a warning in the decision trail.
3. **Anything unsalvageable resolves to the safest fallback for that
   decision type**, never to "guess and continue": malformed JSON, an
   invalid enum value, or an upstream API failure (network error, the
   model not calling the tool at all) all collapse to the same
   `safe_default()` -- for the Reconstruction Agent, that means `action=
   "accept"`: stop, keep the last known-good result.

`llm_agents.py` additionally uses Anthropic's forced tool-calling (a JSON
Schema per agent role in `prompts.py`) rather than "please respond with
JSON" -- a robustness improvement on top of (1)-(3), not a replacement for
them. Every path, mock or LLM, routes through the exact same
`schemas.py` validation; `tests/test_layer4_llm_agents_glue.py` proves this
with stub SDK objects standing in for the real API (no network needed to
verify the glue code itself).

### Bounds table

| Field | Bounds | Source |
|---|---|---|
| `keep_fraction` | 0.3 - 1.0 | `select_frames()`'s own `max(3, ...)` floor + avoiding near-empty selections |
| `min_overall` | 0.0 - 0.5 | `frame_quality.py`'s documented ~0-1 score range |
| `voxel_size` | 0.3 - 10.0 | order-of-magnitude around the 2.0 default across configs seen in this repo |
| `min_consistency` | 0.0 - 0.9 | `depth_estimation.py`'s `consistency` field is a normalized [0,1] score |
| `depth_window` | 1 - 5 (int) | plane-sweep neighbor count; repo default is 2 |
| `n_depths` | 8 - 96 (int) | repo default is 32 |
| `depth_min` / `depth_max` | 0.3-500 / 1-1000 m | generous outer bounds; real constraint is `depth_min < depth_max`, checked after clamping |
| `prune_below` / `densify_above` | 0.0-0.9 / 0.1-1.0 | `confidence.py`'s confidence score is [0,1]; real constraint is `prune_below < densify_above`, checked after clamping |

**These bounds were derived by reading the code, not by running large
sweeps on real footage.** They are believed safe, not proven optimal. In
particular, everything above was exercised on Environment-A-scale synthetic
scenes (dozens of frames, 160-320px-class imagery) -- re-check against a
real 1080p/4K single-pass flight before trusting these ranges there, the
same caveat `frame_overlap.py` already states for its own thresholds.

## The gate-threshold rule, and why it's enforced twice

This project's headline result is that pruning low-confidence points **on
purpose**, even at a cost to completeness, is the right call (the
Spearman confidence-vs-error correlation in `README.md`). An autonomous
retry loop that quietly loosens `prune_below`/`densify_above` to make
completeness numbers look better would be undermining the exact thing it's
supposed to help demonstrate.

This is enforced twice, deliberately redundantly:

- **`mock_agents.MockReconstructionAgent` never proposes changing either
  field.** When low confidence looks like genuine scene difficulty rather
  than a fixable setting, its answer is `flag_and_accept`, not a gate
  change. (`tests/test_layer4_mock_agents.py::
  test_reconstruction_agent_never_touches_prune_or_densify_thresholds`.)
- **The LLM agent path is allowed to propose a gate change** (removing the
  capability entirely seemed worse than making its use maximally visible --
  a genuinely miscalibrated default for an unusual new dataset is a real
  possibility an LLM might reasonably flag). But `RECONSTRUCTION_AGENT_
  SYSTEM_PROMPT` states the constraint explicitly and requires a stated
  reason, and **the orchestrator flags every such change**, from either
  agent type, with a prominent `gate_threshold_changed: true` +
  before/after values in that iteration's record -- never silent, never
  buried in a reasoning string a human might skim past.
  (`tests/test_layer4_orchestrator.py::
  test_orchestrator_never_lets_gate_thresholds_change_silently`.)

## Mock agents vs LLM agents -- the same parallel this repo already draws

`docs/ARCHITECTURE.md` explains Layer 1 as a CPU-testable stand-in for
Layer 3's CUDA/PyTorch implementation: same math, a version that runs
everywhere so the surrounding logic can be verified cheaply. `mock_agents.py`
plays the identical role for `llm_agents.py`: both implement the exact same
`IFrameAgent`/`IReconstructionAgent`/`IEvaluationAgent`/`ITrainingControlAgent`
contracts (`interfaces.py`), so `orchestrator.py`'s retry loop (Layer 1) and
`apply_training_agent_at_checkpoint`'s decision application (Layer 3) are
both fully proven -- in this network- and GPU-disabled sandbox -- using the
mock agents as a substitute, not an approximation.

`llm_agents.py` is implemented and statically reviewed but **not executed
anywhere in this repository's test suite**, for the same reason Layer 3's
CUDA code isn't: the environment it was written in doesn't have what it
needs (there, a GPU; here, outbound network + an API key). This is stated
plainly rather than glossed over, matching this project's own stated
honesty standard in `README.md`.

## Phase status

| Component | Status |
|---|---|
| `perception/frame_overlap.py` | Implemented, CPU-tested (8/8), integration-checked against `select_frames()` output |
| `evaluation/spatial_diagnostics.py` | Implemented, CPU-tested (8/8), integration-checked against a real pipeline run |
| `layer4_agents/schemas.py` | Implemented, CPU-tested (32/32, including adversarial cases and the Training Control Agent's decision type) |
| `layer4_agents/context_builders.py` | Implemented, CPU-tested (11/11), integration-checked against real `PipelineResult`/`FullEvaluation`/`RealEvaluation` objects |
| `layer4_agents/mock_agents.py` | Implemented, CPU-tested (25/25) |
| `layer4_agents/orchestrator.py` | Implemented, CPU-tested end-to-end (8/8) against a real small synthetic dataset, including retry-bound, config-mutation, exception-revert, and history-visibility checks |
| `layer4_agents/prompts.py` | Implemented, CPU-tested (7/7) for shape/bounds consistency |
| `layer4_agents/llm_agents.py` | Implemented, glue logic CPU-tested with stub SDK objects (14/14); the real API path is **not executed anywhere in this repo** -- no network in this sandbox |
| `run_agentic_experiment.py` | Implemented and **actually executed** end-to-end (mock-agent path) -- see worked example below |
| `run_agentic_real_experiment.py` | Implemented and **actually executed** end-to-end against mock DJI-format video+log fixtures (mock-agent path); not yet run against real downloaded footage, same caveat `run_real_experiment.py` already states |
| `train_gpu.py` agent hooks (`apply_training_agent_at_checkpoint`, `_build_layer4_agents`, `_frame_agent_decision_kwargs`, `--use_agents`/`--use_llm`/`--agent_model`) | Implemented; the extracted decision function is CPU-tested against the REAL file (7/7, via scoped dependency stubbing -- see `tests/test_layer3_train_gpu_agent_hooks.py`); the full loop under actual CUDA execution with `--use_agents` on has **not been run** -- needs a real GPU session |

## Worked example (a real run, not illustrative numbers)

`python3 -m layer4_agents.run_agentic_experiment --seed 0 --extent 40.0`
(mock agents, actually executed):

- Frame Agent: raw capture looked healthy, kept the 0.75/0.15 defaults.
- Reconstruction Agent, iteration 0: `spatial_diagnostics` found a real
  weak region (9 points, northwest, 22% low-confidence) in the 944-point
  first attempt. Retried with `depth_window` 2 -> 3 -- a genuine
  configuration adjustment, not a gate change.
- Reconstruction Agent, iteration 1: 1152 points, no weak region left,
  `accept`.
- Evaluation Agent: `accept`, citing the actual control-point RMSE (2.8m,
  inside this repo's own reference band) and surface completeness (3.9%)
  by name.

This is the exact narrative the original suggestion's illustrative example
("the reconstruction is weak around the eastern section... increase frame
overlap... rerun") described -- produced here from this repo's own real
per-point confidence data, on this repo's own algorithms, with no COLMAP
anywhere in the loop.

## Explicitly out of scope for this pass

- **Occlusion/surface completion** ("filling in" unseen sides with agent-
  generated geometry). `docs/STOP_CONDITIONS.md` #1 already governs this
  and is stricter than anything Layer 4 needed to add: build only if
  measured completeness numbers justify it, and never merge invented
  geometry with observed geometry without a persistent, visualized
  confidence-band distinction. `evaluation/occlusion.py`'s existing
  observed/dropped/never-observable breakdown, and the new
  `spatial_diagnostics.py` weak-region summary, are the honest, already-
  built version of "tell the user which parts are unreliable" -- they
  report a gap; they do not fill it. If/when surface completion is
  ever built, it plugs in as a fourth agent behind
  `layer2_interfaces/i_occlusion_completion.py`'s existing interface, and
  should be judged against `STOP_CONDITIONS.md` #1 exactly as written
  there.
- **Letting the Reconstruction Agent re-invoke frame selection.** Frame
  selection happens once, up front. A later retry can adjust reconstruction-
  stage parameters but not `keep_fraction`/`min_overall` again. A
  bidirectional loop (reconstruction quality driving a frame re-selection)
  is a reasonable future extension but adds real complexity (stale overlap
  diagnostics, harder-to-reason-about history) for unclear benefit until
  there's a concrete case where it's needed.
- **A learned Scene Understanding Agent.** See above.

## Running this yourself

```bash
# Layer 1 CPU pipeline, mock agents (no API key needed) -- what's proven to work
# in this repo's own tests:
python3 -m layer4_agents.run_agentic_experiment --out_dir outputs/agentic_run1

python3 -m layer4_agents.run_agentic_real_experiment \
    --video flight.mov --log flight.csv --out_dir outputs/agentic_real_run1

# Layer 3 GPU training (Environment B, real GPU) -- add --use_agents to your existing
# train_gpu.py command, everything else identical:
python3 layer3_gpu/python/train_gpu.py --source real --video flight.mov --log flight.csv \
    --out_dir outputs/agentic_gpu_run1 --max_iterations 20000 --densify_until_iter 15000 \
    --camera_hfov_deg 84 --n_eval_views 3 --use_agents

# Real Claude-backed agents instead of the offline mocks, either script above --
# requires network + ANTHROPIC_API_KEY, and is NOT exercised anywhere in this
# repository (see llm_agents.py):
pip install -r requirements-agents.txt
export ANTHROPIC_API_KEY=...
python3 -m layer4_agents.run_agentic_experiment --out_dir outputs/agentic_run1 --use_llm
python3 layer3_gpu/python/train_gpu.py ... --use_agents --use_llm
```

Before a real `--use_llm` run, check docs.claude.com for the current
recommended model string and any rate-limit/pricing considerations for
however many agent calls your `--max_iterations`/`--checkpoint_every`
setting implies -- `llm_agents.py`'s default (`claude-sonnet-5`) reflects
this repo's information at time of writing and may not stay current.

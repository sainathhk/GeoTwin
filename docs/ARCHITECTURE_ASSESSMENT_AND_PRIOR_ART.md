# Architecture assessment vs. current prior art (2026-09-21)

You asked: are we in the right direction, are we missing anything, how does this compare
to SuGaR / gsplat / anything else out there. Answering all three together, since they're
the same question really. I pulled current papers/repos for this rather than working from
memory -- 3DGS-adjacent work moves fast enough that "what I already know" isn't reliable
here.

## Short answer

Direction: sound. "COLMAP-free 3DGS from a video, using temporal continuity and/or
telemetry instead of full SfM" is an active, real research area right now, not a
simplification you're getting away with for a hackathon -- CF-3DGS (CVPR 2024) and
InstantSplat both do this, and the field is still actively iterating on it. Your per-point/
per-Gaussian **confidence** framework (observation count, view-angle spread, plane-sweep
consistency, frame quality, pose confidence, fused into one number that gates densification
AND is exposed per-point) is a genuine point of difference -- most of what's below tracks
pose and geometry error implicitly through the photometric loss; explicit, multi-signal,
exposed-per-point confidence is not the default anywhere I found it. That's worth keeping
front and center in however you present this.

What's actually missing maps cleanly onto specific, named gaps below -- not "the whole
approach is wrong."

## What today's bug does and doesn't tell you

Nothing here changes the above. The regression fixed today was two functions silently
using the wrong camera intrinsics for some frames -- a plumbing bug, not evidence against
3D Gaussian Splatting, VO-based pose, or the confidence framework. Worth saying plainly
since "the shape, the gradient, all of it looks wrong" is exactly what garbage input
geometry produces regardless of which 3D representation is downstream of it.

## Gap 1: nothing shapes the Gaussians toward the surface during training

This is your "the gaussians don't look right" complaint specifically. `mesh_export.py`
cleans up AFTER training -- opacity/confidence thresholds, floater filtering by anisotropy,
component/debris removal. All real and all now actually wired into the automatic export
(today's other fix), but all of it is still filtering what training already produced. It
can't fix a Gaussian that's the wrong SHAPE to begin with (blobby, tilted off the true
surface, elongated in an arbitrary direction) -- it can only decide to keep or discard it.

**SuGaR** (Guédon & Lepetit, CVPR 2024) targets exactly this. Its core contribution is a
regularization LOSS TERM added during training: it derives a volume density from the
Gaussians under the assumption they're flat and surface-aligned, and penalizes the gap
between that assumed density and the Gaussians' actual one. That pressure during
optimization -- not a post-hoc filter -- is what makes the resulting point set clean
enough for Poisson to give a coherent mesh in minutes instead of the audit's original
"noisy blob." SuGaR also has an optional second stage (bind new Gaussians to the extracted
mesh surface, refine jointly) that's more than you need right now, but the first
contribution -- a flatness/surface-alignment loss term added to `losses.py` -- is a
concrete, scoped addition that targets this complaint directly, in a way no amount of
better export-time filtering can. This is my top pick if you only fix one more thing before
your deadline.

## Gap 2: classical multi-view geometry is fighting your actual constraint (too few views)

Your VO (ORB + essential matrix) and plane-sweep MVS are both classical multi-view
geometry, which fundamentally wants more views with more overlap than a single 14-25s
pass gives you -- that's WHY today's bug (and last session's view-starvation finding)
hurt so much: there's no slack in the system to absorb a bad frame or two.

Two different current research directions respond to that constraint differently, and
they're worth knowing about even if you don't have time to adopt either before judging:

- **InstantSplat** (2024) and **CF-3DGS** (CVPR 2024) jointly optimize camera poses AND
  Gaussians together, end to end, rather than fixing poses from VO first and training
  Gaussians second the way your pipeline does now. Explicitly relevant to something you
  should know: the InstantSplat paper states plainly that CF-3DGS and Nope-NeRF (an
  earlier pose-free NeRF method) "assume accurate focal lengths, limiting their robustness
  in scenarios with focal length uncertainties" -- i.e. a zoom lens mid-clip is a
  recognized open problem in this exact literature, not a sign your approach was naive.
  Last session's audit already recommended a lightweight bundle-adjustment refinement of
  your VO poses (not yet built, still on the list) -- that's the low-risk version of this
  same idea: refine, don't yet fully jointly-optimize.
- **DUSt3R / MASt3R / VGGT** are pretrained transformer models that take unposed images
  and directly output dense point maps + relative pose, no classical feature matching or
  per-frame plane-sweep required. Directly relevant: a December 2025 evaluation
  (Wu, Landgraf, Ulrich & Qin, published in *Geo-spatial Information Science*) ran exactly
  these three models on sparse aerial photogrammetry blocks -- fewer than ten images each
  -- and found "reasonable accuracy and completeness gains up to 50% over COLMAP," with
  VGGT specifically ahead on efficiency, scalability, and pose reliability. Same paper's
  honest caveat: all three degrade on higher-resolution imagery and larger image sets, and
  "transformer-based methods cannot fully replace traditional SfM and MVS, but offer
  promise as complementary approaches, especially in challenging, low-resolution, and
  sparse scenarios" -- which is a fair description of your exact situation (9-22 frames,
  resized to 640x480 for training already). The realistic way to use this given your
  timeline: not a rewrite, but a single VGGT/MASt3R pass on your kept frames as an
  ALTERNATIVE or CROSS-CHECK initialization for VO+plane-sweep, to see whether it gives a
  visibly cleaner seed cloud on this specific footage. That's an experiment measured in
  hours, not a redesign.

## Gap 3: the varying-intrinsics problem you just hit is a known-enough issue that infrastructure exists for it

Not urgent to act on, but worth knowing: **gsplat** (nerfstudio/UC Berkeley/NVIDIA's CUDA
rasterization library) is not a competing end-to-end pipeline -- it's a backend, the kind
of thing your `layer3_gpu/csrc` custom kernel could in principle sit on top of or be
benchmarked against. What IS directly relevant: gsplat has integrated NVIDIA's **3DGUT**
(Unified Gaussian Tracing), which natively supports non-pinhole and varying camera
models -- exactly the class of problem you and I spent today patching by hand, frame by
frame, through five different functions. Not a suggestion to swap rasterizer backends
mid-hackathon (that's a bigger lift than your timeline supports), but if per-frame/varying
intrinsics keeps being a recurring cost, it's the direction the field's own infrastructure
has already moved.

## Gap 4: carried over from last session, still true

Bundle-adjusting the VO poses and a depth-consistency training loss term were both
identified in the previous audit and still aren't built, for the same reason as
before -- both touch pose/training accuracy, which needs a genuine held-out view to
verify against (you have that now, from last session's `--holdout_fraction` fix). Worth
doing once you've confirmed the current fix actually improved held-out PSNR/SSIM/LPIPS
over the 9-camera baseline -- there's no point refining pose accuracy on top of a still-
uncertain foundation.

## If you only have time for one thing before judging

SuGaR's surface-alignment regularization term (Gap 1). It directly targets the complaint
you led with, it's a loss-function addition rather than a pipeline redesign, and unlike
Gaps 2-3 it doesn't require re-deriving your pose/intrinsics handling again.

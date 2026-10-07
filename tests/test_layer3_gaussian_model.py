"""
Tests for layer3_gpu/python/gaussian_model.py. All of this runs on CPU (no CUDA
or diff-gaussian-rasterization needed) since none of GaussianModel's init/
optimizer/prune/densify bookkeeping actually touches rendering -- only
`train_gpu.py`'s render calls need a real GPU + the external rasterizer.

These tests exist because three real bugs were caught by actually running
this code during development (not by reading it): (1) the optimizer's
param_groups not being re-pointed at new Parameter objects after prune/
densify, meaning Adam would silently keep optimizing a stale, disconnected
tensor forever after the first structural change; (2) a lambda signature
mismatch in the padding helper; (3) confidence inheritance for clones being
documented but not actually wired through. All three are guarded here so a
future edit can't silently reintroduce them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layer3_gpu", "python"))

import pytest
torch = pytest.importorskip("torch")

from gaussian_model import GaussianModel, GaussianTrainingConfig


def _toy_model(n=6, seed=0):
    torch.manual_seed(seed)
    positions = torch.rand(n, 3) * 10
    colors = torch.rand(n, 3)
    confidence = torch.rand(n)
    model = GaussianModel(GaussianTrainingConfig(sh_degree=0))
    model.initialize_from_fused_points(positions, colors, confidence, device="cpu")
    return model, positions, colors, confidence


def test_initialize_from_fused_points_shapes():
    model, positions, colors, confidence = _toy_model(6)
    assert model.n_gaussians == 6
    assert model.get_opacity().shape == (6, 1)
    assert model.get_scaling().shape == (6, 3)
    assert model.get_rotation().shape == (6, 4)
    assert model.confidence_stats.obs_count.shape[0] == 6
    assert model.propagator.propagated.shape[0] == 6


def test_low_confidence_points_get_lower_initial_opacity():
    n = 4
    positions = torch.rand(n, 3) * 10
    colors = torch.rand(n, 3)
    confidence = torch.tensor([0.95, 0.05, 0.95, 0.05])
    model = GaussianModel(GaussianTrainingConfig())
    model.initialize_from_fused_points(positions, colors, confidence, device="cpu")
    opacity = model.get_opacity().squeeze()
    assert opacity[0] > opacity[1]
    assert opacity[2] > opacity[3]


def test_optimizer_state_survives_prune():
    """Guards bug #1: after prune, the optimizer's param_groups must be re-pointed at
    the NEW parameter object, with Adam state correctly resized, not left stale."""
    model, *_ = _toy_model(6)
    opt = model.setup_optimizer()
    loss = ((model.xyz - 1.0) ** 2).sum() + model.opacity_logit.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    old_xyz_param = model.xyz
    prune_mask = torch.zeros(6, dtype=torch.bool)
    prune_mask[[1, 3]] = True
    model.prune_points(prune_mask)

    assert model.n_gaussians == 4
    assert model.xyz is not old_xyz_param
    assert model.xyz in opt.state, "optimizer state was not transferred to the new parameter object"
    assert old_xyz_param not in opt.state, "stale old parameter still holds optimizer state"
    assert opt.state[model.xyz]["exp_avg"].shape[0] == 4
    xyz_group = next(g for g in opt.param_groups if g["name"] == "xyz")
    assert xyz_group["params"][0] is model.xyz, "optimizer param_group still references the OLD parameter"

    # training must be able to continue after the structural change
    loss2 = ((model.xyz - 1.0) ** 2).sum()
    loss2.backward()
    opt.step()  # must not raise


def test_clone_and_split_gaussian_count_arithmetic():
    model, *_ = _toy_model(5, seed=1)
    opt = model.setup_optimizer()
    loss = (model.xyz ** 2).sum()
    loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        model.log_scaling[0] = torch.log(torch.tensor(0.1))   # clone candidate
        model.log_scaling[1] = torch.log(torch.tensor(0.1))   # clone candidate
        model.log_scaling[2] = torch.log(torch.tensor(5.0))   # split candidate
        model.log_scaling[3] = torch.log(torch.tensor(5.0))   # split candidate
        model.log_scaling[4] = torch.log(torch.tensor(0.1))   # not densified

    densify_mask = torch.tensor([True, True, True, True, False])
    model.clone_and_split(densify_mask, split_scale_thresh=0.5, extent=1.0)
    # 5 originals + 2 clones (idx 0,1) + (4 split children - 2 removed split parents) = 5+2+2 = 9
    assert model.n_gaussians == 9

    xyz_group = next(g for g in opt.param_groups if g["name"] == "xyz")
    assert xyz_group["params"][0] is model.xyz
    assert opt.state[model.xyz]["exp_avg"].shape[0] == 9
    assert model.confidence_stats.obs_count.shape[0] == 9
    assert model.propagator.propagated.shape[0] == 9

    loss2 = (model.xyz ** 2).sum() + model.opacity_logit.sum()
    loss2.backward()
    opt.step()  # must not raise


def test_clones_inherit_parent_propagated_confidence():
    """Guards bug #3: clones must inherit their parent's propagated observation
    confidence, not silently restart at the new-Gaussian default (0.0)."""
    model, *_ = _toy_model(5, seed=1)
    model.setup_optimizer()
    model.propagator.update(model.xyz.detach(), model.prior_confidence)

    with torch.no_grad():
        model.log_scaling[:] = torch.log(torch.tensor(0.1))  # all clones, simplest case
    densify_mask = torch.tensor([True, True, True, True, False])
    model.clone_and_split(densify_mask, split_scale_thresh=0.5, extent=1.0)

    parents = model.propagator.propagated[:5]
    clones = model.propagator.propagated[5:9]
    assert torch.allclose(clones, parents[:4]), "clones did not inherit parent confidence"


def test_accumulate_view_gradient_only_updates_visible():
    model, *_ = _toy_model(4)
    screenspace = (torch.rand(4, 3) * 2 - 1).requires_grad_(True)
    (screenspace ** 2).sum().backward()
    vis_mask = torch.tensor([True, False, True, True])
    model.accumulate_view_gradient(screenspace, vis_mask)
    mvg = model.mean_view_gradient()
    assert mvg[1] == 0.0
    assert mvg[0] > 0.0 and mvg[2] > 0.0


def test_full_prune_then_densify_orchestration_exactly_as_train_gpu_uses_it():
    """Regression test for a real integration gap caught during development:
    GaussianModel originally had no confidence()/densify_and_prune() of its own
    (only the separate CPU-testable ConfidenceAwareGaussianModel did), so
    train_gpu.py's training loop would have crashed on the first densification
    round. Also guards the prune-before-densify ordering fix: prune_mask and
    densify_mask are computed over the SAME pre-change index space and must be
    re-aligned via `densify_mask[keep_mask]` BEFORE calling clone_and_split,
    not sliced afterward (index semantics shift once clone_and_split removes
    split-parent rows)."""
    torch.manual_seed(5)
    N = 8
    positions = torch.rand(N, 3) * 10
    colors = torch.rand(N, 3)
    # 0,1: low confidence -> prune. 2,3: small scale -> clone. 4,5: large scale -> split. 6,7: untouched.
    confidence = torch.tensor([0.02, 0.03, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    model = GaussianModel(GaussianTrainingConfig(sh_degree=0))
    model.initialize_from_fused_points(positions, colors, confidence, device="cpu")
    opt = model.setup_optimizer()
    loss = (model.xyz ** 2).sum(); loss.backward(); opt.step(); opt.zero_grad()

    with torch.no_grad():
        model.opacity_logit[:] = 3.0
        model.log_scaling[2] = torch.log(torch.tensor(0.1))
        model.log_scaling[3] = torch.log(torch.tensor(0.1))
        model.log_scaling[4] = torch.log(torch.tensor(5.0))
        model.log_scaling[5] = torch.log(torch.tensor(5.0))
        model.log_scaling[[0, 1, 6, 7]] = torch.log(torch.tensor(0.1))

    grad_accum = torch.tensor([0, 0, 0.001, 0.001, 0.001, 0.001, 0, 0], dtype=torch.float32)
    prune_mask, densify_mask, conf = model.densify_and_prune(grad_accum, confidence_prune_below=0.12,
                                                               confidence_densify_above=0.6)
    assert prune_mask.tolist() == [True, True, False, False, False, False, False, False]
    assert densify_mask.tolist() == [False, False, True, True, True, True, False, False]

    keep_mask = ~prune_mask
    densify_mask_after_prune = densify_mask[keep_mask]
    assert densify_mask_after_prune.tolist() == [True, True, True, True, False, False]

    model.prune_points(prune_mask)
    assert model.n_gaussians == 6
    model.clone_and_split(densify_mask_after_prune, split_scale_thresh=0.5, extent=1.0)
    assert model.n_gaussians == 10  # 6 survivors + 2 clones + (4 split children - 2 removed parents)

    xyz_group = next(g for g in opt.param_groups if g["name"] == "xyz")
    assert xyz_group["params"][0] is model.xyz
    assert opt.state[model.xyz]["exp_avg"].shape[0] == 10
    assert model.confidence_stats.obs_count.shape[0] == 10
    assert model.propagator.propagated.shape[0] == 10

    loss2 = (model.xyz ** 2).sum() + model.opacity_logit.sum()
    loss2.backward()
    opt.step()  # must not raise


def test_full_state_save_and_reload_round_trip():
    """Guards a real gap found in practice: the first successful 10,000-iteration
    training run (real footage, real T4 GPU) only saved position+color .ply
    checkpoints, meaning the trained model was unrecoverable -- no re-rendering,
    no PSNR/SSIM against held-out frames, no continued training -- once the
    process exited. This test guards that a saved checkpoint can actually be
    reloaded into a fully working GaussianModel (same parameters, same
    confidence, and importantly: still trainable, not just inspectable)."""
    import tempfile, os
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "layer3_gpu", "python"))
    from train_gpu import _save_full_state, load_full_state

    torch.manual_seed(0)
    n = 10
    model = GaussianModel(GaussianTrainingConfig(sh_degree=0))
    model.initialize_from_fused_points(torch.rand(n, 3) * 10, torch.rand(n, 3), torch.rand(n), device="cpu")

    with tempfile.TemporaryDirectory() as d:
        _save_full_state(model, d, 100)
        path = os.path.join(d, "full_state_iter100.pt")
        assert os.path.exists(path)

        reloaded = load_full_state(path, device="cpu")
        assert reloaded.n_gaussians == n
        assert torch.allclose(reloaded.xyz, model.xyz)
        assert torch.allclose(reloaded.get_opacity(), model.get_opacity())
        assert torch.allclose(reloaded.confidence(), model.confidence())
        assert reloaded.confidence_stats.obs_count.shape[0] == n

        opt = reloaded.setup_optimizer()
        loss = (reloaded.xyz ** 2).sum() + reloaded.opacity_logit.sum()
        loss.backward()
        opt.step()  # must not raise -- reloaded model must still be trainable, not just inspectable


def test_full_state_round_trip_preserves_accumulated_confidence_not_just_prior():
    """`test_full_state_save_and_reload_round_trip` above passes even WITHOUT saving
    confidence_stats, because it saves a model that was never actually trained --
    `confidence_stats.n_accum` is all-zero on both sides, so `confidence()` on BOTH
    the original and the reload falls through to `prior_confidence` alone (see
    `blended_confidence`'s `has_obs = stats.n_accum > 0` gate in
    confidence_gaussian_model.py), which is then trivially equal either way.

    This test simulates what actually happens after real training iterations --
    confidence_stats.update() has real accumulated evidence -- which is exactly the
    case the original _save_full_state silently lost: it saved
    xyz/features/opacity/prior_confidence but not confidence_stats itself, so
    reloading built a fresh, all-zero GaussianConfidenceStats and confidence()
    silently fell back to the (here, deliberately much lower) creation-time prior
    instead of what training actually observed. Caught while wiring confidence-gated
    mesh export through a reloaded checkpoint (export_obj.py), not by inspection."""
    import tempfile, os
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "layer3_gpu", "python"))
    from train_gpu import _save_full_state, load_full_state

    torch.manual_seed(1)
    n = 8
    # Deliberately low prior: makes it obvious whether reload used it verbatim (bug)
    # or blended it with the high-consistency evidence accumulated below (fixed).
    prior = torch.full((n,), 0.05)
    model = GaussianModel(GaussianTrainingConfig(sh_degree=0))
    model.initialize_from_fused_points(torch.rand(n, 3) * 10, torch.rand(n, 3), prior, device="cpu")

    idx = torch.arange(n)
    for i in range(5):
        # view_angle DELIBERATELY varies across calls (0.02 -> 0.14 rad): the angle-spread
        # term in GaussianConfidenceStats.confidence() specifically rewards having seen a
        # Gaussian from a spread of angles, not just repeatedly from the same one -- an
        # identical angle every call scores angle_score=0 (zero spread) and tanks the
        # geometric-mean formula almost to zero regardless of how good the other 4 signals
        # are, which would make this test's "blended clearly exceeds prior" margin razor-thin
        # instead of robust. Verified numerically before relying on it here.
        view_angle = torch.full((n,), 0.02 + i * 0.03)
        consistency = torch.full((n,), 0.95)
        model.confidence_stats.update(idx, view_angle, consistency, quality=0.9, pose_conf=0.9)

    live_confidence = model.confidence()
    assert bool((live_confidence > prior).all()), \
        "test setup should make blended confidence clearly exceed the low prior"

    with tempfile.TemporaryDirectory() as d:
        _save_full_state(model, d, 200)
        reloaded = load_full_state(os.path.join(d, "full_state_iter200.pt"), device="cpu")

        assert reloaded.confidence_stats.n_accum.sum().item() == model.confidence_stats.n_accum.sum().item()
        assert torch.allclose(reloaded.confidence(), live_confidence, atol=1e-5), (
            "reload silently fell back to prior_confidence instead of the accumulated, "
            "trained confidence -- confidence_stats did not survive the round trip")


def test_split_train_holdout_cameras_disjoint_and_covers_all():
    """Guards a real gap found in practice: the first two successful training runs
    only ever rendered TRAINING views for checkpoint PNGs -- a good render there
    proves memorization, not reconstruction. This is the fix's core invariant:
    train and held-out sets must be disjoint and together cover every camera."""
    import sys as _sys, os as _os
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "layer3_gpu", "python"))
    from train_gpu import split_train_holdout_cameras, TrainingCamera

    cams = [TrainingCamera(pose=None, K=None, gt_image=torch.zeros(3, 4, 4), frame_quality=1.0,
                            pose_confidence=1.0) for _ in range(60)]
    for n in [10, 45, 60]:
        train, held = split_train_holdout_cameras(cams[:n], holdout_fraction=0.15)
        assert len(train) + len(held) == n
        assert held is not train


def test_split_train_holdout_cameras_degenerate_single_camera_falls_back():
    import sys as _sys, os as _os
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "layer3_gpu", "python"))
    from train_gpu import split_train_holdout_cameras, TrainingCamera

    cams = [TrainingCamera(pose=None, K=None, gt_image=torch.zeros(3, 4, 4), frame_quality=1.0,
                            pose_confidence=1.0)]
    train, held = split_train_holdout_cameras(cams, holdout_fraction=0.15)
    assert held is train, "with only 1 camera, must fall back to identical lists"


def test_camera_bridge_matches_reference_3dgs_transform_order():
    """The rasterizer consumes transposed matrices, so the combined matrix is
    ``world_view.T @ projection.T``, not the reversed product."""
    from layer1_cpu_sandbox.synthetic.camera_model import CameraPose, Intrinsics
    from train_gpu import _camera_to_rasterizer_matrices, TrainingCamera
    import numpy as np

    K = Intrinsics(width=640, height=480, fx=500.0, fy=510.0, cx=320.0, cy=240.0)
    pose = CameraPose(position=np.array([4.0, -2.0, 7.0]), R_wc=np.eye(3))
    cam = TrainingCamera(pose=pose, K=K, gt_image=torch.zeros(3, 480, 640),
                         frame_quality=1.0, pose_confidence=1.0)
    viewmatrix, full_proj, tanx, tany = _camera_to_rasterizer_matrices(cam, "cpu")

    world_view = torch.eye(4)
    world_view[:3, 3] = -torch.tensor([4.0, -2.0, 7.0])
    projection = torch.zeros(4, 4)
    projection[0, 0] = 1.0 / tanx
    projection[1, 1] = 1.0 / tany
    projection[2, 2] = 500.0 / (500.0 - 0.05)
    projection[2, 3] = -(500.0 * 0.05) / (500.0 - 0.05)
    projection[3, 2] = 1.0

    assert torch.allclose(viewmatrix, world_view.T)
    assert torch.allclose(full_proj, world_view.T @ projection.T)
    assert not torch.allclose(full_proj, world_view @ projection)


def test_downward_mapping_filter_excludes_horizon_views_but_preserves_source_ids():
    from train_gpu import filter_downward_mapping_frames
    from layer1_cpu_sandbox.real_data.real_dataset_builder import RealDroneDataset, RealFrame
    from layer1_cpu_sandbox.synthetic.camera_model import CameraPose, Intrinsics
    import numpy as np

    K = Intrinsics.from_fov(16, 12, 84.0)
    down = CameraPose(position=np.zeros(3), R_wc=np.diag([1., -1., -1.]))
    horizon = CameraPose(position=np.zeros(3), R_wc=np.eye(3))
    frames = [RealFrame(idx=i, t=float(i), rgb=np.zeros((12, 16, 3), dtype=np.uint8), pose=down,
                        gps_lat=0., gps_lon=0., position_residual_m=0.) for i in range(8)]
    frames.append(RealFrame(idx=99, t=9., rgb=np.zeros((12, 16, 3), dtype=np.uint8), pose=horizon,
                            gps_lat=0., gps_lon=0., position_residual_m=0.))
    dataset = RealDroneDataset(frames=frames, K=K, K_is_assumed=True, origin_lat=0., origin_lon=0.,
                               video_info=None, log_format_detected="test", matched_log_offset_s=0.)
    filtered, rejected = filter_downward_mapping_frames(dataset, max_forward_z=-0.5)
    assert [f.idx for f in filtered.frames] == list(range(8))
    assert rejected == [99]


def test_densification_budget_requires_fresh_observation_and_keeps_top_gradients():
    from train_gpu import _limit_densify_candidates

    candidates = torch.tensor([True, True, True, True, False])
    gradients = torch.tensor([0.2, 0.9, 0.5, 0.8, 1.0])
    observations = torch.tensor([2.0, 1.0, 3.0, 4.0, 20.0])
    selected = _limit_densify_candidates(candidates, gradients, observations,
                                         min_observations=2, max_new=2)
    # Index 1 has too little independent evidence; among 0,2,3 select 3 and 2.
    assert selected.tolist() == [False, False, True, True, False]

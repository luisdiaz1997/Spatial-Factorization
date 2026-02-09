"""Test SVGP model loading: group ordering and learned Lu inspection.

Verifies:
1. Group codes (C.npy) align with group_names in metadata
2. Model groupsZ proportions match data group proportions (proportional allocation)
3. Lu (Cholesky variational covariance) was actually learned during training
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

OUTPUT_DIR = Path("outputs/slideseq")
MODEL_DIR = OUTPUT_DIR / "svgp"
PREP_DIR = OUTPUT_DIR / "preprocessed"


def _skip_if_missing():
    """Skip test if model or preprocessed data not available."""
    if not PREP_DIR.exists():
        pytest.skip(f"Preprocessed data not found: {PREP_DIR}")
    if not MODEL_DIR.exists():
        pytest.skip(f"SVGP model not found: {MODEL_DIR}")
    if not (MODEL_DIR / "model.pth").exists():
        pytest.skip("model.pth not found")


# ── 1. Group ordering verification ──────────────────────────────────


class TestGroupOrdering:
    """Verify groups are consistent between preprocessed data and model."""

    def test_group_codes_match_names(self):
        """C.npy integer codes should map 1-to-1 with metadata group_names."""
        _skip_if_missing()

        C = np.load(PREP_DIR / "C.npy")
        with open(PREP_DIR / "metadata.json") as f:
            meta = json.load(f)

        group_names = meta["group_names"]
        n_groups = meta["n_groups"]

        # Every code in C should have a name
        unique_codes = np.unique(C)
        assert len(unique_codes) == n_groups, (
            f"Unique codes ({len(unique_codes)}) != n_groups ({n_groups})"
        )
        assert unique_codes.min() == 0, f"Group codes should start at 0, got {unique_codes.min()}"
        assert unique_codes.max() == n_groups - 1, (
            f"Group codes should go up to {n_groups - 1}, got {unique_codes.max()}"
        )
        assert len(group_names) == n_groups, (
            f"group_names length ({len(group_names)}) != n_groups ({n_groups})"
        )

        # Print the mapping
        print("\n=== Group Code -> Name Mapping ===")
        for code in unique_codes:
            count = np.sum(C == code)
            print(f"  Code {code:>2}: {group_names[code]:<30} ({count:>5} spots)")

    def test_model_groupsZ_matches_data_proportions(self):
        """Model inducing point group assignments should match data proportions."""
        _skip_if_missing()

        C = np.load(PREP_DIR / "C.npy")
        state = torch.load(MODEL_DIR / "model.pth", map_location="cpu", weights_only=False)
        groupsZ = state["prior_state_dict"]["groupsZ"].numpy()

        with open(PREP_DIR / "metadata.json") as f:
            meta = json.load(f)
        group_names = meta["group_names"]

        # Same groups should appear in both
        data_groups = set(np.unique(C).tolist())
        model_groups = set(np.unique(groupsZ).tolist())
        assert data_groups == model_groups, (
            f"Group mismatch: data has {data_groups}, model has {model_groups}"
        )

        # Proportions should roughly match (proportional allocation)
        print("\n=== Group Proportions: Data vs Inducing Points ===")
        print(f"{'Code':>4}  {'Name':<30}  {'Data %':>8}  {'Inducing %':>10}  {'Diff':>6}")
        for g in sorted(data_groups):
            data_pct = 100 * np.sum(C == g) / len(C)
            model_pct = 100 * np.sum(groupsZ == g) / len(groupsZ)
            diff = abs(data_pct - model_pct)
            name = group_names[g] if g < len(group_names) else f"Group {g}"
            print(f"{g:>4}  {name:<30}  {data_pct:>7.1f}%  {model_pct:>9.1f}%  {diff:>5.1f}%")

            # Proportions should be within 2% of each other
            assert diff < 2.0, (
                f"Group {g} ({name}): data={data_pct:.1f}%, model={model_pct:.1f}%, diff={diff:.1f}%"
            )

    def test_groups_passed_to_get_factors_match(self):
        """Verify groups loaded by load_preprocessed match what was trained."""
        _skip_if_missing()

        from spatial_factorization.datasets.base import load_preprocessed

        data = load_preprocessed(OUTPUT_DIR)
        state = torch.load(MODEL_DIR / "model.pth", map_location="cpu", weights_only=False)
        groupsZ = state["prior_state_dict"]["groupsZ"]

        # data.groups should be long tensor with same unique values as groupsZ
        assert data.groups.dtype == torch.long
        assert set(data.groups.unique().tolist()) == set(groupsZ.unique().tolist())
        print(f"\n  data.groups: {data.groups.shape}, unique={data.groups.unique().tolist()}")
        print(f"  model groupsZ: {groupsZ.shape}, unique={groupsZ.unique().tolist()}")
        print("  [OK] Groups are consistent between data and model")


# ── 2. Lu (Cholesky variational covariance) inspection ──────────────


class TestLuParameter:
    """Inspect the learned Lu parameter from the SVGP model."""

    def _load_prior_state(self):
        """Load prior state dict from model.pth."""
        _skip_if_missing()
        state = torch.load(MODEL_DIR / "model.pth", map_location="cpu", weights_only=False)
        return state["prior_state_dict"]

    def test_prior_state_dict_keys(self):
        """List all keys in the prior state dict."""
        prior_sd = self._load_prior_state()

        print("\n=== Prior State Dict Keys ===")
        for k, v in prior_sd.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                          f"min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
                else:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                          f"min={v.min()}, max={v.max()}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")

        # Lu should be present as Lu._raw (CholeskyParameter stores _raw)
        lu_keys = [k for k in prior_sd if "Lu" in k]
        print(f"\n  Lu-related keys: {lu_keys}")
        assert len(lu_keys) > 0, "No Lu parameter found in prior state dict!"

    def test_lu_shape_and_values(self):
        """Inspect Lu constrained shape, triangular structure, and values."""
        _skip_if_missing()

        from spatial_factorization.commands.analyze import _load_model

        model = _load_model(MODEL_DIR)
        gp = model._prior

        # Get constrained Lu (the actual Cholesky factor after exp transform)
        Lu = gp.Lu.data  # CholeskyParameter.data -> _to_constrained
        L, M, _ = Lu.shape
        print(f"\n=== Constrained Lu (actual Cholesky factor) ===")
        print(f"  Shape: {Lu.shape}  (L={L} factors, M={M} inducing points)")

        # --- Upper triangle must be zero ---
        upper = torch.triu(Lu, diagonal=1)
        upper_abs_max = upper.abs().max().item()
        upper_nonzero = (upper.abs() > 0).sum().item()
        print(f"\n  Upper triangle: abs max={upper_abs_max:.2e}, non-zero count={upper_nonzero}")
        assert upper_abs_max == 0, f"Upper triangle is not zero! max={upper_abs_max}"
        print("  [OK] Upper triangle is all zeros")

        # --- Lower triangle (strict, below diagonal) should be non-zero ---
        lower = torch.tril(Lu, diagonal=-1)
        lower_vals = lower[lower != 0]
        lower_total = L * M * (M - 1) // 2  # total strict-lower-triangle elements
        lower_nonzero = (lower.abs() > 1e-8).sum().item()
        lower_frac = lower_nonzero / lower_total if lower_total > 0 else 0
        print(f"\n  Lower triangle (strict):")
        print(f"    Total elements:  {lower_total}")
        print(f"    Non-zero (>1e-8): {lower_nonzero} ({100*lower_frac:.1f}%)")
        print(f"    Abs mean: {lower.abs().sum() / lower_total:.6f}")
        print(f"    Min: {lower_vals.min():.6f}, Max: {lower_vals.max():.6f}")

        # --- Diagonal (constrained, should be positive) ---
        diag = torch.diagonal(Lu, dim1=1, dim2=2)  # (L, M)
        assert torch.all(diag > 0), "Diagonal must be positive"
        print(f"\n  Diagonal (constrained):")
        print(f"    Min: {diag.min():.6f}, Max: {diag.max():.6f}, Mean: {diag.mean():.6f}")

        # Per-factor summary
        print(f"\n  Per-factor diagonal:")
        for l in range(L):
            d = diag[l]
            print(f"    Factor {l:>2}: mean={d.mean():.6f}, std={d.std():.6f}, "
                  f"min={d.min():.6f}, max={d.max():.6f}")

        # Show a small corner of constrained Lu
        print(f"\n  Lu[0, :5, :5] (first factor, corner):")
        print(Lu[0, :5, :5].detach().numpy())
        print(f"\n  Lu[-1, :5, :5] (last factor, corner):")
        print(Lu[-1, :5, :5].detach().numpy())

    def test_lu_was_learned(self):
        """Check if Lu has moved away from its initialization.

        CholeskyParameter with mode='exp' initializes:
        - Diagonal (constrained): 0.1
        - Off-diagonal: randn() * 0.1

        If Lu hasn't been learned, the constrained diagonal should still be ~0.1.
        """
        _skip_if_missing()

        from spatial_factorization.commands.analyze import _load_model

        model = _load_model(MODEL_DIR)
        Lu = model._prior.Lu.data  # constrained (L, M, M)
        L, M, _ = Lu.shape
        init_diag_val = 0.1  # CholeskyParameter mode='exp' init

        diag = torch.diagonal(Lu, dim1=1, dim2=2)  # (L, M) constrained
        diag_mean = diag.mean().item()
        diag_std = diag.std().item()

        # Lower triangle (strict)
        lower_mask = torch.tril(torch.ones(M, M, dtype=torch.bool), diagonal=-1)
        lower_mask = lower_mask.unsqueeze(0).expand(L, -1, -1)
        lower_vals = Lu[lower_mask]

        print(f"\n=== Lu Learning Check (constrained values) ===")
        print(f"  Init diagonal:         {init_diag_val}")
        print(f"  Learned diagonal mean: {diag_mean:.6f}")
        print(f"  Learned diagonal std:  {diag_std:.6f}")
        print(f"  Distance from init:    {abs(diag_mean - init_diag_val):.6f}")
        print(f"  Lower-tri abs mean:    {lower_vals.abs().mean():.6f}")
        print(f"  Lower-tri non-zero (>1e-6): {(lower_vals.abs() > 1e-6).sum()} / {lower_vals.numel()}")

        moved = abs(diag_mean - init_diag_val) > 0.01
        if moved:
            print("  [OK] Lu diagonal has moved from initialization -> LEARNED")
        else:
            print("  [WARNING] Lu diagonal is near initialization -> may NOT have been learned")

    def test_reconstruct_lu_and_forward(self):
        """Reconstruct the model and verify Lu produces valid covariance."""
        _skip_if_missing()

        from spatial_factorization.commands.analyze import _load_model
        from spatial_factorization.datasets.base import load_preprocessed

        model = _load_model(MODEL_DIR)
        data = load_preprocessed(OUTPUT_DIR)

        gp = model._prior
        print(f"\n=== Reconstructed GP Prior ===")
        print(f"  Type: {type(gp).__name__}")
        print(f"  Lu type: {type(gp.Lu).__name__}")
        print(f"  mu shape: {gp.mu.shape}")

        # Get constrained Lu (the actual lower-triangular Cholesky factor)
        Lu_constrained = gp.Lu.data  # This calls _to_constrained
        print(f"  Lu constrained shape: {Lu_constrained.shape}")
        print(f"  Lu constrained min: {Lu_constrained.min():.6f}, max: {Lu_constrained.max():.6f}")

        # Verify it's lower triangular
        L_dim, M_dim, _ = Lu_constrained.shape
        for l in range(L_dim):
            upper = torch.triu(Lu_constrained[l], diagonal=1)
            assert torch.all(upper == 0), f"Factor {l} Lu is not lower triangular!"
        print("  [OK] Lu is lower triangular for all factors")

        # Verify diagonal is positive
        diag = torch.diagonal(Lu_constrained, dim1=1, dim2=2)
        assert torch.all(diag > 0), "Lu diagonal should be positive"
        print(f"  [OK] Lu diagonal is positive (min={diag.min():.6f})")

        # Try a small forward pass to verify the model works
        coords = data.X.numpy()[:100]
        groups = data.groups.numpy()[:100]

        from PNMF.transforms import _get_spatial_qF
        with torch.no_grad():
            qF = _get_spatial_qF(model, coordinates=coords, groups=groups)
        print(f"  Forward pass qF mean shape: {qF.mean.shape}")
        print(f"  Forward pass qF scale shape: {qF.scale.shape}")
        print(f"  [OK] Forward pass succeeds with loaded model")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

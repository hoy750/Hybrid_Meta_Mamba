from models.deep.mamba import mamba2_in_projection_dim, validate_mamba2_cuda_alignment


def test_project_mamba_configs_are_cuda_stride_aligned():
    assert mamba2_in_projection_dim(
        hidden_dim=128,
        state_dim=128,
        expand=4,
        head_dim=64,
        ngroups=1,
    ) % 8 == 0
    assert mamba2_in_projection_dim(
        hidden_dim=64,
        state_dim=64,
        expand=4,
        head_dim=32,
        ngroups=1,
    ) % 8 == 0


def test_invalid_mamba2_alignment_is_reported_before_training():
    try:
        validate_mamba2_cuda_alignment(
            hidden_dim=128,
            state_dim=128,
            expand=2,
            head_dim=64,
            ngroups=1,
        )
    except ValueError as exc:
        assert "projection dimension must be a multiple of 8" in str(exc)
    else:
        raise AssertionError("invalid Mamba2 config should raise ValueError")

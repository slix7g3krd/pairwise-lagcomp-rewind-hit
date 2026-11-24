import pytest
import gameplay.lagcomp_rewind_hit as mod

CASES = [
    ("snapshot_by_ts", (1,)),
    ("clamp_interp", (1,)),
    ("isolate_request", (1,)),
    ("validate_handle", (1,)),
    ("drop_stale_seq", (1,)),
    ("record_snapshot", (1,)),
    ("rewind", (1,)),
    ("resolve_hit", (1,)),
    ("confirm", (1,)),
    ("release", (1,)),
    ("validate_hit", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

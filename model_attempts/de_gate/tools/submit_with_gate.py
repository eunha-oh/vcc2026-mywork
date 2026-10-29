"""scripts/vcc2026_submission.py 를 그대로 부르되, 불러온 게이트 모델의 추론 모드만 바꾼다 (제출 스크립트는 고치지 않는다).

사용: python tools/de_gate/submit_with_gate.py --gate-mode hard <vcc2026_submission.py 인자 그대로>
--gate-mode 를 주지 않으면 학습 설정의 모드(기본 soft_norm)를 쓴다. 게이트가 없는 모델에는 쓸 수 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import vcc2026_submission as submission  # noqa: E402


def main(argv):
    mode, threshold = None, None
    rest = []
    it = iter(argv)
    for arg in it:
        if arg == "--gate-mode":
            mode = next(it)
        elif arg == "--gate-hard-threshold":
            threshold = float(next(it))
        else:
            rest.append(arg)
    original = submission.load_model

    def load_model(*args, **kwargs):
        model, blob, cfg, commit = original(*args, **kwargs)
        if mode is not None:
            if not hasattr(model, "gate_mode"):
                raise SystemExit("--gate-mode 는 DE 게이트 모델에만 쓸 수 있다")
            model.gate_mode = mode
        if threshold is not None:
            model.gate_hard_threshold = threshold
        if hasattr(model, "gate_mode"):
            print(f"DE gate mode used: {model.gate_mode} (hard threshold {model.gate_hard_threshold})", flush=True)
        return model, blob, cfg, commit

    submission.load_model = load_model
    submission.main(rest)


if __name__ == "__main__":
    main(sys.argv[1:])

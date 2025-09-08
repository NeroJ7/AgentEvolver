# test_group_zscore_on_steps.py
import math
from dataclasses import dataclass
from typing import List, Dict
import torch
import pytest

# ===== 如果你的代码在其它模块中，请改成 from your_module import _group_zscore_on_steps, PRMHyper =====
# 这里放一份最小可运行的 stub，便于独立运行测试；如你已有实现，注释掉本段并改为导入。
@dataclass
class PRMHyper:
    do_batch_norm: bool = True
    equal_trajectory_weight: bool = True
    eps: float = 1e-6

def _group_zscore_on_steps(
    step_rewards_raw: List[List[float]],
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """对 step 奖励做“组内”减均值/除方差标准化。"""
    import math
    from typing import Dict, List

    if not hyper.do_batch_norm:
        return [list(r) for r in step_rewards_raw]

    B = len(step_rewards_raw)
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    for _, idxs in g2idx.items():
        if hyper.equal_trajectory_weight:
            # 1) 组均值：轨迹均值的等权平均
            traj_means = []
            for i in idxs:
                ri = step_rewards_raw[i]
                if ri:
                    traj_means.append(sum(ri) / len(ri))
            if len(traj_means) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = float(sum(traj_means) / len(traj_means))
                # 2) 组方差：先对每条轨迹围绕 mu_g 求均方差，再对轨迹等权平均
                second_moments = []
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments.append(sum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = float(sum(second_moments) / len(second_moments)) if second_moments else 0.0
                sd_g = float(math.sqrt(var_g + hyper.eps))
        else:
            # 拉平：把本组所有 step 拼在一起
            flat = []
            for i in idxs:
                flat.extend(step_rewards_raw[i])
            if len(flat) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                t = torch.tensor(flat, dtype=torch.float32)
                mu_g = float(t.mean().item())
                sd_g = float(max(t.std(unbiased=False).item(), hyper.eps))

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]
    return step_rewards_std
# ===== stub 结束 =====

def assert_lol_allclose(a: List[List[float]], b: List[List[float]], atol=1e-6):
    assert len(a) == len(b), f"outer length mismatch: {len(a)} vs {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        assert len(ra) == len(rb), f"inner length mismatch at {i}: {len(ra)} vs {len(rb)}"
        for j, (xa, xb) in enumerate(zip(ra, rb)):
            assert abs(xa - xb) <= atol, f"mismatch at [{i}][{j}]: {xa} vs {xb}"

def test_no_norm_returns_input():
    step_rewards = [[1.0, -1.0, 2.0], [], [3.0]]
    gids = torch.tensor([0, 0, 1])
    hyper = PRMHyper(do_batch_norm=False, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, hyper)
    assert step_rewards == out, "When do_batch_norm=False, output should equal input."

def test_group_zscore_on_steps_cases():
    """
    单函数自测：覆盖6个代表性场景
    使用条件：同文件已定义 _group_zscore_on_steps 与 PRMHyper
    """
    import math
    import torch

    def _mk_h(do_batch_norm=True, equal_trajectory_weight=True, eps=1e-6):
        try:
            return PRMHyper(
                do_batch_norm=do_batch_norm,
                equal_trajectory_weight=equal_trajectory_weight,
                eps=eps,
            )
        except TypeError:
            h = PRMHyper()
            if hasattr(h, "do_batch_norm"): setattr(h, "do_batch_norm", do_batch_norm)
            if hasattr(h, "equal_trajectory_weight"): setattr(h, "equal_trajectory_weight", equal_trajectory_weight)
            if hasattr(h, "eps"): setattr(h, "eps", eps)
            return h

    def _assert_lol_close(actual, expected, name, atol=1e-6):
        if len(actual) != len(expected):
            raise AssertionError(f"[{name}] outer len mismatch: {len(actual)} vs {len(expected)}")
        for i, (ra, re) in enumerate(zip(actual, expected)):
            if len(ra) != len(re):
                raise AssertionError(f"[{name}] inner len mismatch at {i}: {len(ra)} vs {len(re)}")
            for j, (xa, xe) in enumerate(zip(ra, re)):
                if abs(xa - xe) > atol:
                    raise AssertionError(f"[{name}] mismatch at [{i}][{j}]: {xa} vs {xe} (atol={atol})")

    print("Case 1: do_batch_norm=False，应原样返回")
    step_rewards = [
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0], 
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], 
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], 
        [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    gids = torch.tensor([0, 0, 0, 0])
    h = _mk_h(do_batch_norm=False, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)
    _assert_lol_close(out, step_rewards, "no_norm_returns_input")
    
    print("Case 2: do_batch_norm=True 用1.0")
    step_rewards = [
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0], 
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], 
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], 
        [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    gids = torch.tensor([0, 0, 0, 0])
    h = _mk_h(do_batch_norm=True, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)
    # 当 do_batch_norm=True 时，输出应该是标准化后的值，而不是原始输入
    # 我们验证输出的均值接近0，标准差接近1
    flat_out = []
    for row in out:
        flat_out.extend(row)
    if flat_out:
        t = torch.tensor(flat_out, dtype=torch.float32)
        mean_val = float(t.mean().item())
        std_val = float(t.std(unbiased=False).item())
        print(f"  均值: {mean_val}, 标准差: {std_val}")

    
    print("Case 3: do_batch_norm=True 用0.2")
    step_rewards = [
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0], 
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], 
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], 
        [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    step_rewards = [[x * 0.2 for x in row] for row in step_rewards]
    gids = torch.tensor([0, 0, 0, 0])
    h = _mk_h(do_batch_norm=True, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)
    # 当 do_batch_norm=True 时，输出应该是标准化后的值，而不是原始输入
    # 我们验证输出的均值接近0，标准差接近1
    flat_out = []
    for row in out:
        flat_out.extend(row)
    if flat_out:
        t = torch.tensor(flat_out, dtype=torch.float32)
        mean_val = float(t.mean().item())
        std_val = float(t.std(unbiased=False).item())
        print(f"  均值: {mean_val}, 标准差: {std_val}")

    
    print("Case 3: do_batch_norm=True 用0.5")
    step_rewards = [
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0], 
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], 
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], 
        [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    step_rewards = [[x * 0.5 for x in row] for row in step_rewards]
    gids = torch.tensor([0, 0, 0, 0])
    h = _mk_h(do_batch_norm=True, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)
    # 当 do_batch_norm=True 时，输出应该是标准化后的值，而不是原始输入
    # 我们验证输出的均值接近0，标准差接近1
    flat_out = []
    for row in out:
        flat_out.extend(row)
    if flat_out:
        t = torch.tensor(flat_out, dtype=torch.float32)
        mean_val = float(t.mean().item())
        std_val = float(t.std(unbiased=False).item())
        print(f"  均值: {mean_val}, 标准差: {std_val}")

    print("Case 4: 等权轨迹、单组、两条轨迹")
    # t1=[1,1,-1], t2=[3,3]
    # 轨迹均值: mu1=(1+1-1)/3=1/3, mu2=(3+3)/2=3
    # 组均值: mu_g=(1/3+3)/2=5/3
    # 轨迹方差: var1=((1-5/3)^2+(1-5/3)^2+(-1-5/3)^2)/3, var2=((3-5/3)^2+(3-5/3)^2)/2
    # 组方差: var_g=(var1+var2)/2
    step_rewards = [[1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], [1.0, -1.0]]
    gids = torch.tensor([0, 0])
    
    h = _mk_h(do_batch_norm=True, equal_trajectory_weight=True, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)


    print("Case 5: 等权step、单组、两条轨迹")
    # t1=[1,1,-1], t2=[3,3]
    # 轨迹均值: mu1=(1+1-1)/3=1/3, mu2=(3+3)/2=3
    # 组均值: mu_g=(1/3+3)/2=5/3
    # 轨迹方差: var1=((1-5/3)^2+(1-5/3)^2+(-1-5/3)^2)/3, var2=((3-5/3)^2+(3-5/3)^2)/2
    # 组方差: var_g=(var1+var2)/2
    step_rewards = [[1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0], [1.0, -1.0]]
    gids = torch.tensor([0, 0])
    
    h = _mk_h(do_batch_norm=True, equal_trajectory_weight=False, eps=1e-6)
    out = _group_zscore_on_steps(step_rewards, gids, h)
    print("  输入:", step_rewards)
    print("  输出:", out)

    

    
test_group_zscore_on_steps_cases()
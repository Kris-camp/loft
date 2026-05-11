import types
from tqdm import tqdm
from peft.utils import _get_submodules
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F


class LOFTLayer(nn.Module):
    def __init__(self, size, orth=True, use_cayley_neumann=True, num_cayley_neumann_terms=5):
        super().__init__()
        self.size = size
        self.orth = orth
        # self.mag_b = mag_b
        # self.mag_a = mag_a
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms

        if self.orth:
            self.a_params = nn.Parameter(torch.zeros((size * (size - 1)) // 2))
            self.register_buffer("indices_lower", torch.tril_indices(size, size, -1))
        else:
            self.a_params = nn.Parameter(torch.eye(size))

        # self.mag_vector_b = nn.Parameter(torch.ones(size)) if self.mag_b else None
        # self.mag_vector_a = nn.Parameter(torch.ones(size)) if self.mag_a else None

    def forward(self, input):
        # returns input @ Q^T
        if self.orth:
            A = torch.zeros((self.size, self.size), device=self.a_params.device, dtype=self.a_params.dtype)
            A[self.indices_lower[0], self.indices_lower[1]] = self.a_params
            A = A - A.t()
            I = torch.eye(self.size, device=A.device, dtype=A.dtype)

            if self.use_cayley_neumann:
                t = self.num_cayley_neumann_terms
                if t <= 0:
                    Q = I - A
                else:
                    negA = -A
                    R = I.clone()
                    P = negA
                    for _ in range(t):
                        R.add_(P, alpha=2.0)
                        P = P @ negA
                    R.add_(P)
                    Q = R
            else:
                Q = torch.linalg.solve(I + A, I - A, left=False)
        else:
            Q = self.a_params

        # if self.mag_b and self.mag_a:
        #     Q = torch.diag(self.mag_vector_b) @ Q @ torch.diag(self.mag_vector_a)
        # elif self.mag_b:
        #     Q = torch.diag(self.mag_vector_b) @ Q
        # elif self.mag_a:
        #     Q = Q @ torch.diag(self.mag_vector_a)

        return input @ Q.t()


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


# ---------- 核心：固定 P 的 right-multiply 版本 ----------
class LOFTFixedPLayer(nn.Module):
    """
    Implements x -> x U^T where U = I + Pr^T (R - I) Pr, Pr fixed (r x d_in).
    With row-batch convention, right-multiply means we apply U^T on the RIGHT of x.

    xU^T = x + (x Pr^T) (R^T - I) Pr
    """
    def __init__(
        self,
        Pr: torch.Tensor,  # (r, d_in), orthonormal rows
        orth=True,
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        # keep strict orthogonality by default:
        # mag_b=False,
        # mag_a=False,
    ):
        super().__init__()
        assert Pr.dim() == 2
        r, d_in = Pr.shape
        self.r = r
        self.d_in = d_in

        self.register_buffer("Pr", Pr)         # (r, d_in)
        self.register_buffer("PrT", Pr.t())    # (d_in, r)
        self.register_buffer("Ir", torch.eye(r, device=Pr.device, dtype=Pr.dtype))

        # R parameterization (size=r)
        self.R_layer = LOFTLayer(
            size=r,
            orth=orth,
            # mag_b=mag_b,
            # mag_a=mag_a,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Pr = self.Pr.to(dtype=x.dtype)
        PrT = self.PrT.to(dtype=x.dtype)

        # tmp = x Pr^T
        tmp = x @ PrT               # (..., r)

        # tmpR = tmp R^T  (LOFTLayer returns input @ R^T)
        tmpR = self.R_layer(tmp)    # (..., r)

        # tmp (R^T - I)
        delta_r = tmpR - tmp        # (..., r)

        # delta = (x Pr^T)(R^T - I) Pr
        delta = delta_r @ Pr        # (..., d_in)

        return x + delta


# ---------- 从 LOFTFixedPLayer 里把 R 矩阵“显式”还原出来（用于 merge delta weight） ----------
def _extract_R_matrix_from_loftlayer(layer: LOFTLayer) -> torch.Tensor:
    """
    Return Q matrix (r x r) used inside LOFTLayer. For strict orth, set mag_a=mag_b=False.
    """
    size = layer.size
    a_params = layer.a_params
    dtype = a_params.dtype
    device = a_params.device

    if layer.orth:
        idx = layer.indices_lower
        A = torch.zeros((size, size), device=device, dtype=dtype)
        A[idx[0], idx[1]] = a_params
        A = A - A.t()
        I = torch.eye(size, device=device, dtype=dtype)

        if layer.use_cayley_neumann:
            t = layer.num_cayley_neumann_terms
            if t <= 0:
                Q = I - A
            else:
                negA = -A
                Acc = I.clone()
                P = negA
                for _ in range(t):
                    Acc.add_(P, alpha=2.0)
                    P = P @ negA
                Acc.add_(P)
                Q = Acc
        else:
            Q = torch.linalg.solve(I + A, I - A, left=False)
    else:
        Q = a_params

    # if layer.mag_b and layer.mag_a:
    #     Q = torch.diag(layer.mag_vector_b) @ Q @ torch.diag(layer.mag_vector_a)
    # elif layer.mag_b:
    #     Q = torch.diag(layer.mag_vector_b) @ Q
    # elif layer.mag_a:
    #     Q = Q @ torch.diag(layer.mag_vector_a)

    return Q


# ---------- merge 用：ΔW = W(U - I) ----------
def get_delta_weight_loft_fixedP(self, adapter: str) -> torch.Tensor:
    """
    For W -> WU, U = I + Pr^T(R - I)Pr.
    So ΔW = W(U - I) = W Pr^T (R - I) Pr.
    Return in stored orientation consistent with PEFT.
    """
    layer: LOFTFixedPLayer = self.loft_fixedP[adapter]
    device = self.weight.device
    dtype = self.weight.dtype
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    W_used = transpose(self.weight, self.fan_in_fan_out)  # (out, in)
    if cast_to_fp32:
        W_used = W_used.float()

    Pr = layer.Pr.to(device=W_used.device, dtype=W_used.dtype)     # (r, in)
    PrT = layer.PrT.to(device=W_used.device, dtype=W_used.dtype)   # (in, r)

    R = _extract_R_matrix_from_loftlayer(layer.R_layer).to(device=W_used.device, dtype=W_used.dtype)  # (r, r)
    Ir = layer.Ir.to(device=W_used.device, dtype=W_used.dtype)
    R_minus_I = R - Ir

    # W Pr^T : (out, r)
    WPrT = W_used @ PrT
    # (out, r) @ (r, r) @ (r, in) -> (out, in)
    delta_used = (WPrT @ R_minus_I) @ Pr

    out = transpose(delta_used, self.fan_in_fan_out)
    if cast_to_fp32:
        out = out.to(dtype=dtype)
    return out


# ---------- forward：先 xU^T 再 linear（严格对应 X(UW)^T = (XU^T)W^T） ----------
def forward_loft_fixedP(self, x: torch.Tensor):
    previous_dtype = x.dtype
    adapter = self.active_adapter[0] if isinstance(self.active_adapter, (list, tuple)) else self.active_adapter

    # 没插就回退
    if not hasattr(self, "loft_fixedP") or adapter not in self.loft_fixedP:
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    # 关 adapter：直接 base linear
    if self.disable_adapters:
        if getattr(self, "merged", False):
            self.unmerge()
        y = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return y.to(previous_dtype)

    # merged：权重已经变成 WU 了
    if getattr(self, "merged", False):
        y = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return y.to(previous_dtype)

    # 未 merge：先做 x <- xU^T，再做 linear（bias 不被旋转）
    x2 = self.loft_fixedP[adapter](x.to(self.weight.dtype))
    y = F.linear(x2, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    return y.to(previous_dtype)


# ---------- Pr 的构造：V_r^T / grad_svd / random / perm ----------
@torch.no_grad()
def build_Pr_from_matrix_right(M_used: torch.Tensor, r: int) -> torch.Tensor:
    """
    Generic right-subspace builder from an arbitrary matrix M_used: return top-r RIGHT singular
    subspace as Pr = V_r^T with shape (r, d_in).
    """
    m, d_in = M_used.shape
    k = min(m, d_in)
    if r > k:
        raise ValueError(f"rank r={r} > min(shape)={k} for matrix shape={tuple(M_used.shape)}")
    _, _, Vh = torch.linalg.svd(M_used.float(), full_matrices=False)
    return Vh[:r, :].to(dtype=M_used.dtype)


@torch.no_grad()
def build_Pr_from_skew_wg_right(W_used: torch.Tensor, G_used: torch.Tensor, r: int) -> torch.Tensor:
    """
    Build P_r from the top-r invariant subspace of

        S = skew(W^T G) = (W^T G - G^T W) / 2

    where W_used and G_used are both shaped (out, in).

    Practical implementation:
    Since S is real skew-symmetric, we recover its dominant real invariant subspace
    via the top eigenspace of

        C = S^T S = -S^2

    which is symmetric PSD. We then return Pr = V_r^T with shape (r, in).
    """
    if W_used.shape != G_used.shape:
        raise ValueError(
            f"W_used.shape={tuple(W_used.shape)} and G_used.shape={tuple(G_used.shape)} must match."
        )

    _, d_in = W_used.shape
    if r > d_in:
        raise ValueError(f"rank r={r} > in_features={d_in}")

    if r % 2 != 0:
        print(
            f"[LOFT][wg_skew] warning: r={r} is odd. "
            "Skew-symmetric invariant subspaces are naturally even-dimensional; "
            "proceeding with the top-r eigenspace of S^T S."
        )

    Wf = W_used.float()
    Gf = G_used.float()

    S = 0.5 * (Wf.t() @ Gf - Gf.t() @ Wf)   # (in, in), skew-symmetric
    C = S.t() @ S                           # symmetric PSD

    evals, V = torch.linalg.eigh(C)         # ascending
    idx = torch.argsort(evals, descending=True)[:r]
    Pr = V[:, idx].t().contiguous()         # (r, in)

    return Pr.to(dtype=W_used.dtype)

@torch.no_grad()
def build_Pr_from_weight_right(W_used: torch.Tensor, r: int, mode: str = "svd", seed: int = 42) -> torch.Tensor:
    """
    W_used: (out, in). We need top RIGHT singular subspace => V_r^T.
    Return Pr: (r, in) with orthonormal rows.
    """
    out, d_in = W_used.shape
    if r > d_in:
        raise ValueError(f"rank r={r} > in_features={d_in}")

    if mode == "perm":
        g = torch.Generator(device=W_used.device)
        g.manual_seed(seed)
        idx = torch.randperm(d_in, generator=g, device=W_used.device)[:r]
        Pr = torch.zeros((r, d_in), device=W_used.device, dtype=W_used.dtype)
        Pr[torch.arange(r, device=W_used.device), idx] = 1
        return Pr

    if mode == "random":
        g = torch.Generator(device=W_used.device)
        g.manual_seed(seed)
        M = torch.randn((d_in, r), generator=g, device=W_used.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(M, mode="reduced")   # (d_in, r) columns orthonormal
        return Q.t().to(dtype=W_used.dtype)         # (r, d_in) rows orthonormal

    if mode == "svd":
        # W = U S V^T  => Pr = V_r^T
        return build_Pr_from_matrix_right(W_used, r=r)

    raise ValueError(f"Unknown Pr init mode: {mode}")


# ---------- 注入函数（对齐你现在 create_and_insert 的风格） ----------
def create_and_insert_loft_fixedP(
    model,
    peft_config,
    Pr_init: str = "svd",     # "svd" | "grad_svd" | "wg_skew" | "random" | "perm"
    grad_map=None,
    seed: int = 0,
    orth: bool = True,
    use_cayley_neumann: bool = True,
    num_cayley_neumann_terms: int = 5,
    freeze_lora_AB: bool = True ,
):
    is_found = False
    key_list = [key for key, _ in model.named_modules()]
    assert not isinstance(peft_config.target_modules, str)

    seed_it = seed

    print(f"Inserting loft fixed-P module (right-multiply) with Pr_init={Pr_init}.")
    for key in tqdm(key_list):
        if not any(key.endswith(tk) for tk in peft_config.target_modules):
            continue

        _, target, _ = _get_submodules(model, key)

        if not isinstance(target, peft.tuners.lora.Linear):
            raise NotImplementedError("Only peft.tuners.lora.Linear targets are supported (same as your PSOFT patch).")

        is_found = True

        # patch
        target.forward = types.MethodType(forward_loft_fixedP, target)
        target.get_delta_weight = types.MethodType(get_delta_weight_loft_fixedP, target)

        # build Pr on THIS layer: using RIGHT singular subspace of W
        W_used = transpose(target.weight, target.fan_in_fan_out).detach()  # (out, in)
        if Pr_init in {"grad_svd", "wg_skew"}:
            if grad_map is None:
                raise ValueError(f"Pr_init={Pr_init!r} requires grad_map.")

            if key in grad_map:
                G_used = grad_map[key]
            else:
                matches = [g for n, g in grad_map.items() if key.endswith(n) or n.endswith(key)]
                if len(matches) != 1:
                    raise KeyError(f"Cannot uniquely match grad for layer {key}")
                G_used = matches[0]

            G_used = G_used.to(device=W_used.device, dtype=torch.float32)

            if Pr_init == "grad_svd":
                Pr = build_Pr_from_matrix_right(G_used, r=peft_config.r).to(
                    device=W_used.device, dtype=W_used.dtype
                )
            else:
                Pr = build_Pr_from_skew_wg_right(
                    W_used=W_used,
                    G_used=G_used,
                    r=peft_config.r,
                ).to(device=W_used.device, dtype=W_used.dtype)
        else:
            Pr = build_Pr_from_weight_right(
                W_used,
                r=peft_config.r,
                mode=Pr_init,
                seed=seed_it,
            )
        seed_it += 1

        target.loft_fixedP = nn.ModuleDict({
            "default": LOFTFixedPLayer(
                Pr=Pr,
                orth=orth,
                use_cayley_neumann=use_cayley_neumann,
                num_cayley_neumann_terms=num_cayley_neumann_terms,
                # mag_b=False,
                # mag_a=False,
            )
        })

        # 如果你只是想训练 R，不想训练 LoRA A/B，就冻结它们
        if freeze_lora_AB and hasattr(target, "lora_A") and "default" in target.lora_A:
            target.lora_A["default"].weight.requires_grad = False
        if freeze_lora_AB and hasattr(target, "lora_B") and "default" in target.lora_B:
            target.lora_B["default"].weight.requires_grad = False

    if not is_found:
        raise ValueError(f"Target modules {peft_config.target_modules} not found in the base model.")

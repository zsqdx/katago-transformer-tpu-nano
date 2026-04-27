"""Precision test for inv_quarter_sandwich (Shampoo) and polar_express (Muon)."""

import argparse
import torch

from optimizers import inv_quarter_sandwich, polar_express


def make_spd(size, cond_number, device):
    """Generate a random symmetric positive definite matrix with given condition number."""
    # QR on CPU to avoid broken CUDA linalg
    A = torch.randn(size, size, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    eigvals = torch.logspace(0, torch.log10(torch.tensor(cond_number, dtype=torch.float64)),
                             size, dtype=torch.float64)
    spd = Q @ torch.diag(eigvals) @ Q.T
    return spd.to(device)


def inv_quarter_root_exact(M, dtype=torch.float64):
    """Compute M^{-1/4} via eigendecomposition (fp64, on CPU)."""
    M_cpu = M.to(dtype).cpu()
    eigvals, eigvecs = torch.linalg.eigh(M_cpu)
    eigvals = eigvals.clamp(min=1e-12)
    result = eigvecs @ torch.diag(eigvals ** (-0.25)) @ eigvecs.T
    return result.to(M.device)


def test_shampoo(device):
    """Test inv_quarter_sandwich precision across sizes and condition numbers."""
    sizes = [(32, 32), (128, 128), (512, 512), (128, 64), (64, 256)]
    cond_numbers = [10, 100, 1000]

    print(f"{'size':>12s}  {'cond':>6s}  {'rel_frob_err':>14s}  {'max_abs_err':>14s}  {'cosine_sim':>12s}")
    print("-" * 72)

    for (m, n) in sizes:
        for cond in cond_numbers:
            torch.manual_seed(42)

            L = make_spd(m, cond, device)
            R = make_spd(n, cond, device)
            M = torch.randn(m, n, device=device, dtype=torch.float64)

            # Ground truth in fp64
            L_inv4 = inv_quarter_root_exact(L)
            R_inv4 = inv_quarter_root_exact(R)
            exact = L_inv4 @ M @ R_inv4

            # Approximation via Newton-Schulz (fp32/TF32)
            approx = inv_quarter_sandwich(L.float(), M.float(), R.float())
            approx = approx.double()

            # Metrics
            rel_frob = (approx - exact).norm() / exact.norm()
            max_abs = (approx - exact).abs().max()
            cosine = torch.nn.functional.cosine_similarity(
                approx.reshape(1, -1), exact.reshape(1, -1)
            ).item()

            print(f"{str((m,n)):>12s}  {cond:>6d}  {rel_frob.item():>14.6e}  {max_abs.item():>14.6e}  {cosine:>12.8f}")


def test_muon(device):
    """Test polar_express precision across sizes and condition numbers."""
    configs = [
        # (m, n, cond)
        (32, 32, 10), (32, 32, 100), (32, 32, 1000),
        (128, 128, 10), (128, 128, 100), (128, 128, 1000),
        (512, 512, 10), (512, 512, 100), (512, 512, 1000),
        # m > n
        (256, 128, 10), (256, 128, 100), (256, 128, 1000),
        # m < n
        (128, 256, 10), (128, 256, 100), (128, 256, 1000),
    ]

    print(f"{'size':>12s}  {'cond':>6s}  {'rel_frob_err':>14s}  {'max_abs_err':>14s}  {'cosine_sim':>12s}")
    print("-" * 72)

    for (m, n, cond) in configs:
        torch.manual_seed(42)

        # Generate matrix with specified condition number via SVD (on CPU)
        U, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.float64))
        V, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        k = min(m, n)
        sing_vals = torch.logspace(0, torch.log10(torch.tensor(cond, dtype=torch.float64)),
                                   k, dtype=torch.float64)
        S = torch.zeros(m, n, dtype=torch.float64)
        S[:k, :k] = torch.diag(sing_vals)
        G = (U @ S @ V.T).to(device)

        # Ground truth: polar factor U @ V^T (thin SVD on CPU)
        U_svd, _, Vh_svd = torch.linalg.svd(G.cpu().double(), full_matrices=False)
        exact = (U_svd @ Vh_svd).to(device)

        # Approximation via Newton-Schulz (bf16)
        approx = polar_express(G.float())  # polar_express internally converts to bf16
        approx = approx.double()

        # Metrics
        rel_frob = (approx - exact).norm() / exact.norm()
        max_abs = (approx - exact).abs().max()
        cosine = torch.nn.functional.cosine_similarity(
            approx.reshape(1, -1), exact.reshape(1, -1)
        ).item()

        print(f"{str((m,n)):>12s}  {cond:>6d}  {rel_frob.item():>14.6e}  {max_abs.item():>14.6e}  {cosine:>12.8f}")


def main():
    parser = argparse.ArgumentParser(description="Precision test for Shampoo/Muon Newton-Schulz iterations")
    parser.add_argument("--mode", choices=["shampoo", "muon"], required=True,
                        help="shampoo: test inv_quarter_sandwich; muon: test polar_express")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    device = "cuda"

    if args.mode == "shampoo":
        print("=== inv_quarter_sandwich precision (fp32/TF32 vs fp64 ground truth) ===\n")
        test_shampoo(device)
    else:
        print("=== polar_express precision (bf16 vs fp64 ground truth) ===\n")
        test_muon(device)


if __name__ == "__main__":
    main()

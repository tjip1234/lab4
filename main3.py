import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.stats import pearsonr

# ---------------------------------------------------
# 1) FILENAMES (All in ~0–100 scale)
# ---------------------------------------------------
REF_FILE     = "reference-scale.csv"    # 1‑hexanol, in ~0–100
TEST_FILE    = "test-scale.csv"         # Test data, might not strictly min=0 / max=100
HEXANE_FILE  = "hexane.csv"             # Hexane, already 0–100 or close
TWO_HEX_FILE = "2hexanol.csv"           # 2‑hexanol, already 0–100 or close
THF_FILE     = "thf.csv"                # THF, 0–100 as well

# ---------------------------------------------------
# 2) LOAD CSVs
# ---------------------------------------------------
ref_df     = pd.read_csv(REF_FILE,     header=None, names=["wavelength", "transmittance"])
test_df    = pd.read_csv(TEST_FILE,    header=None, names=["wavelength", "transmittance"])
hexane_df  = pd.read_csv(HEXANE_FILE,  header=None, names=["wavelength", "transmittance"])
two_hex_df = pd.read_csv(TWO_HEX_FILE, header=None, names=["wavelength", "transmittance"])
thf_df     = pd.read_csv(THF_FILE,     header=None, names=["wavelength", "transmittance"])

# Sort by wavelength
for df in [ref_df, test_df, hexane_df, two_hex_df, thf_df]:
    df.sort_values(by="wavelength", inplace=True)

# ---------------------------------------------------
# 3) DETERMINE REFERENCE MIN/MAX & SCALE ONLY THE TEST
# ---------------------------------------------------
ref_min = ref_df["transmittance"].min()
ref_max = ref_df["transmittance"].max()

test_min = test_df["transmittance"].min()
test_max = test_df["transmittance"].max()

# Linear transform test to match reference range
# new_value = ((value - test_min)/(test_max - test_min)) * (ref_max - ref_min) + ref_min
def scale_test_to_ref(value):
    return ((value - test_min) / (test_max - test_min)) * (ref_max - ref_min) + ref_min

test_df["transmittance_scaled"] = scale_test_to_ref(test_df["transmittance"])

# For clarity, other components remain "transmittance" as is
# (assuming they’re already on the same 0–100 scale as the reference)

# ---------------------------------------------------
# 4) DEFINE OVERLAPPING RANGE & COMMON GRID
# ---------------------------------------------------
LOW_WL  = 500
HIGH_WL = 4000

full_min = max(
    ref_df["wavelength"].min(),
    test_df["wavelength"].min(),
    hexane_df["wavelength"].min(),
    two_hex_df["wavelength"].min(),
    thf_df["wavelength"].min()
)
full_max = min(
    ref_df["wavelength"].max(),
    test_df["wavelength"].max(),
    hexane_df["wavelength"].max(),
    two_hex_df["wavelength"].max(),
    thf_df["wavelength"].max()
)

# also clamp to 500–4000
full_min = max(full_min, LOW_WL)
full_max = min(full_max, HIGH_WL)

common_wl = np.linspace(full_min, full_max, 2000)

# ---------------------------------------------------
# 5) INTERPOLATE
# ---------------------------------------------------
# 1-hexanol (reference)
ref_interp = np.interp(common_wl, ref_df["wavelength"], ref_df["transmittance"])

# Test (scaled)
test_interp = np.interp(common_wl, test_df["wavelength"], test_df["transmittance_scaled"])

# Hexane
hexane_interp = np.interp(common_wl, hexane_df["wavelength"], hexane_df["transmittance"])

# 2-hexanol
two_hex_raw = np.interp(common_wl, two_hex_df["wavelength"], two_hex_df["transmittance"])

# THF
thf_interp = np.interp(common_wl, thf_df["wavelength"], thf_df["transmittance"])

# ---------------------------------------------------
# 6) MODIFY 2-HEXANOL ABOVE 2000 => REF
# ---------------------------------------------------
two_hex_modified = np.where(common_wl > 2000, ref_interp, two_hex_raw)

# ---------------------------------------------------
# 7) BUILD DESIGN MATRIX & RUN NNLS
# ---------------------------------------------------
X = np.column_stack([ref_interp, two_hex_modified, hexane_interp, thf_interp])
Y = test_interp

coeffs, _ = nnls(X, Y)  # ensures nonnegative solutions

# Normalize so sum=1
sum_coeff = np.sum(coeffs)
if sum_coeff > 0:
    coeffs /= sum_coeff

a = coeffs[0]  # fraction 1‑hexanol
b = coeffs[1]  # fraction 2‑hexanol
c = coeffs[2]  # fraction hexane
d = coeffs[3]  # fraction THF

# ---------------------------------------------------
# 8) RECONSTRUCT & EVALUATE
# ---------------------------------------------------
recon = a*ref_interp + b*two_hex_modified + c*hexane_interp + d*thf_interp

def compute_rms(u, v):
    return np.sqrt(np.mean((u - v)**2))

r_val, p_val = pearsonr(Y, recon)
rms_val = compute_rms(Y, recon)

print("\n[SINGLE-PASS NNLS - 4 Components]")
print("Scaled only the Test data to match the Reference's min/max.")
print(f"  1‑hexanol fraction: {a*100:.2f}%")
print(f"  2‑hexanol fraction: {b*100:.2f}%")
print(f"  Hexane fraction:    {c*100:.2f}%")
print(f"  THF fraction:       {d*100:.2f}%")
print(f"  Pearson r:          {r_val:.4f} (p-value={p_val:.4e})")
print(f"  RMS error:          {rms_val:.4f}")

# ---------------------------------------------------
# 9) PLOT
# ---------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(common_wl, Y, label="Test (Scaled to Ref)", color="red", linewidth=2)
plt.plot(common_wl, recon, label="Reconstructed", color="black", linestyle="--", linewidth=2)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance (0–100 scale)")
plt.title("Single-Pass NNLS Fit (Scale Test Only)")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("final_single_pass_4component.png")
plt.show()

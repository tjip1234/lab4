import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import nnls

# ====================================================
# 1. LOAD & PREPARE DATA
# ====================================================

# Filenames (edit as needed)
REF_FILE     = "reference.csv"     # 1-hexanol reference
TEST_FILE    = "test.csv"          # Your test sample
HEXANE_FILE  = "hexane.csv"        # Hexane
TWO_HEX_FILE = "2hexanol.csv"      # 2‑hexanol

# Load data, name columns
ref_df     = pd.read_csv(REF_FILE,     header=None, names=["wavelength", "transmittance"])
test_df    = pd.read_csv(TEST_FILE,    header=None, names=["wavelength", "transmittance"])
hexane_df  = pd.read_csv(HEXANE_FILE,  header=None, names=["wavelength", "transmittance"])
two_hex_df = pd.read_csv(TWO_HEX_FILE, header=None, names=["wavelength", "transmittance"])

# Convert any 0–100% data to 0–1 (assume reference & hexane are already 0–1)
test_df["transmittance"]    = test_df["transmittance"]    / 100.0
two_hex_df["transmittance"] = two_hex_df["transmittance"] / 100.0

# Sort by wavelength
for df in [ref_df, test_df, hexane_df, two_hex_df]:
    df.sort_values(by="wavelength", inplace=True)

# ====================================================
# 2. SCALE TEST, HEXANE, AND 2-HEXANOL TO THE REFERENCE RANGE
# ====================================================
def scale_to_ref(df, ref_min, ref_max):
    """Linearly scale df['transmittance'] so its min/max match ref_min/ref_max."""
    orig_min = df["transmittance"].min()
    orig_max = df["transmittance"].max()
    df["transmittance_scaled"] = (
        (df["transmittance"] - orig_min) / (orig_max - orig_min)
    ) * (ref_max - ref_min) + ref_min
    return df

ref_min = ref_df["transmittance"].min()  # e.g. ~0
ref_max = ref_df["transmittance"].max()  # e.g. ~0.913

test_df   = scale_to_ref(test_df,   ref_min, ref_max)
hexane_df = scale_to_ref(hexane_df, ref_min, ref_max)
two_hex_df= scale_to_ref(two_hex_df,ref_min, ref_max)

# ====================================================
# 3. HELPER FUNCTIONS
# ====================================================
def extract_region(df, col="transmittance_scaled", wmin=2000, wmax=4000):
    """Extract region [wmin, wmax] from df, using the given column (scaled or original)."""
    return df[(df["wavelength"] >= wmin) & (df["wavelength"] <= wmax)].copy()

def compute_rms(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# ====================================================
# 4. REGION 1: 2000–4000 cm⁻¹ (1-hexanol + hexane)
# ====================================================
region1_min = 2000
region1_max = 4000

ref_r1    = extract_region(ref_df,    "transmittance",       region1_min, region1_max)
hex_r1    = extract_region(hexane_df, "transmittance_scaled",region1_min, region1_max)
test_r1   = extract_region(test_df,   "transmittance_scaled",region1_min, region1_max)

wl_r1 = np.linspace(
    max(ref_r1["wavelength"].min(), hex_r1["wavelength"].min(), test_r1["wavelength"].min()),
    min(ref_r1["wavelength"].max(), hex_r1["wavelength"].max(), test_r1["wavelength"].max()),
    500
)

ref_r1_int  = np.interp(wl_r1, ref_r1["wavelength"],  ref_r1["transmittance"])
hex_r1_int  = np.interp(wl_r1, hex_r1["wavelength"],  hex_r1["transmittance_scaled"])
test_r1_int = np.interp(wl_r1, test_r1["wavelength"], test_r1["transmittance_scaled"])

# Solve for a0, c0 with ordinary least squares
X_r1 = np.column_stack([ref_r1_int, hex_r1_int])
Y_r1 = test_r1_int
(a0, c0), residuals_r1, _, _ = np.linalg.lstsq(X_r1, Y_r1, rcond=None)

# Normalize
sum_ac = a0 + c0
a0 /= sum_ac
c0 /= sum_ac

test_r1_recon = a0 * ref_r1_int + c0 * hex_r1_int
r1_rms = compute_rms(test_r1_int, test_r1_recon)

print("[REGION 1: 2000–4000 cm⁻¹]")
print(f"  => 1‑hexanol fraction (a0): {a0*100:.2f}%")
print(f"  => Hexane fraction (c0):    {c0*100:.2f}%")
print(f"  => RMS error in region 1:   {r1_rms:.4f}")

plt.figure(figsize=(8,4))
plt.plot(wl_r1, test_r1_int, label="Test (Scaled)", color="red")
plt.plot(wl_r1, test_r1_recon, label="Reconstructed", color="black", linestyle="--")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title("Region 1 (2000–4000 cm⁻¹) Fit: Test vs. (1‑hexanol + Hexane)")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("FINAL-region1_fit.png")
plt.show()

# ====================================================
# 5. REGION 2: 500–2000 cm⁻¹ => sub out portion of 1‑hexanol for 2‑hexanol
# ====================================================
region2_min = 500
region2_max = 2000

ref_r2    = extract_region(ref_df,    "transmittance",       region2_min, region2_max)
hex_r2    = extract_region(hexane_df, "transmittance_scaled",region2_min, region2_max)
two_r2    = extract_region(two_hex_df,"transmittance_scaled",region2_min, region2_max)
test_r2   = extract_region(test_df,   "transmittance_scaled",region2_min, region2_max)

wl_r2 = np.linspace(
    max(ref_r2["wavelength"].min(), hex_r2["wavelength"].min(), two_r2["wavelength"].min(), test_r2["wavelength"].min()),
    min(ref_r2["wavelength"].max(), hex_r2["wavelength"].max(), two_r2["wavelength"].max(), test_r2["wavelength"].max()),
    500
)

ref_r2_int  = np.interp(wl_r2, ref_r2["wavelength"],  ref_r2["transmittance"])
hex_r2_int  = np.interp(wl_r2, hex_r2["wavelength"],  hex_r2["transmittance_scaled"])
two_r2_int  = np.interp(wl_r2, two_r2["wavelength"],  two_r2["transmittance_scaled"])
test_r2_int = np.interp(wl_r2, test_r2["wavelength"], test_r2["transmittance_scaled"])

# Baseline from region1 fractions
baseline_r2 = a0*ref_r2_int + c0*hex_r2_int
residual_r2 = test_r2_int - baseline_r2
diff_2hex   = two_r2_int - ref_r2_int

num = np.sum(residual_r2 * diff_2hex)
den = np.sum(diff_2hex * diff_2hex)
b_raw = 0 if den == 0 else (num / den)
# clamp b to [0, a0]
b = max(0, min(b_raw, a0))

a0_prime = a0 - b
c0_prime = c0
b_prime  = b

test_r2_fit = a0_prime*ref_r2_int + c0_prime*hex_r2_int + b_prime*two_r2_int
r2_rms = compute_rms(test_r2_int, test_r2_fit)

print("[REGION 2: 500–2000 cm⁻¹] ~ Subbing out part of 1‑hexanol for 2‑hexanol")
print(f"   => a0' (1‑hexanol): {a0_prime*100:.2f}%")
print(f"   => b'  (2‑hexanol): {b_prime*100:.2f}%")
print(f"   => c0' (hexane):    {c0_prime*100:.2f}%")
print(f"   => RMS error region 2: {r2_rms:.4f}")

plt.figure(figsize=(8,4))
plt.plot(wl_r2, test_r2_int, label="Test (Scaled)", color="red")
plt.plot(wl_r2, test_r2_fit, label="Reconstructed", color="black", linestyle="--")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title("Region 2 (500–2000 cm⁻¹): Subbing out portion of 1‑hexanol for 2‑hexanol")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("FINAL-region2_fit.png")
plt.show()

# ====================================================
# 6. FULL RANGE RECONSTRUCTION (2-step approach)
#    => Replace 2‑hexanol > 2000 cm⁻¹ with 1‑hexanol data
# ====================================================
# final a0, b, c0 from region steps
a_final = a0_prime  # (1-hex in region2)
b_final = b_prime   # (2-hex)
c_final = c0_prime  # (hexane)

# Overlapping range among all four
full_min = max(ref_df["wavelength"].min(), test_df["wavelength"].min(),
               hexane_df["wavelength"].min(), two_hex_df["wavelength"].min())
full_max = min(ref_df["wavelength"].max(), test_df["wavelength"].max(),
               hexane_df["wavelength"].max(), two_hex_df["wavelength"].max())
wl_full = np.linspace(full_min, full_max, 1000)

ref_full     = np.interp(wl_full, ref_df["wavelength"],    ref_df["transmittance"])
test_full    = np.interp(wl_full, test_df["wavelength"],   test_df["transmittance_scaled"])
hex_full     = np.interp(wl_full, hexane_df["wavelength"], hexane_df["transmittance_scaled"])
two_full_raw = np.interp(wl_full, two_hex_df["wavelength"],two_hex_df["transmittance_scaled"])

# Replace 2‑hexanol data above 2000 with 1‑hexanol
two_full_modified = np.where(wl_full > 2000, ref_full, two_full_raw)
full_recon_2step = a_final * ref_full + b_final * two_full_modified + c_final * hex_full

r_2step, p_2step = pearsonr(test_full, full_recon_2step)
rms_2step = compute_rms(test_full, full_recon_2step)

print("[FULL RANGE ANALYSIS - 2-step method]")
print(f"  1-hexanol fraction: {a_final*100:.2f}%")
print(f"  2-hexanol fraction: {b_final*100:.2f}%")
print(f"  Hexane fraction:    {c_final*100:.2f}%")
print(f"  Pearson correlation: r = {r_2step:.4f}, p-value = {p_2step:.4e}")
print(f"  RMS error:           {rms_2step:.4f}")

plt.figure(figsize=(10,6))
plt.plot(wl_full, test_full, label="Test Spectrum (Scaled)", color="red", linewidth=2)
plt.plot(wl_full, full_recon_2step, label="Reconstructed (2-step)", color="black", linestyle="--", linewidth=2)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title("Full Range Reconstruction (2-step approach)")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("FINAL-full_reconstruction_2step.png")
plt.show()

# ====================================================
# 7. SINGLE FULL-RANGE FIT (NNLS) => 3 Components
#    - Also supplement 2‑hex data above 2000 with 1‑hex
# ====================================================
# We'll do a single pass, entire overlap. Then define a matrix X = [ref, two_hex_mod, hex]
two_hex_modified_full = np.where(wl_full > 2000, ref_full, two_full_raw)

# Construct design matrix
X_full = np.column_stack([ref_full, two_hex_modified_full, hex_full])
Y_full = test_full

# Nonnegative least squares => ensure no negative fractions
sol, rnorm = nnls(X_full, Y_full)
a_nnls, b_nnls, c_nnls = sol

# Normalize so that a_nnls + b_nnls + c_nnls = 1
abc_sum = a_nnls + b_nnls + c_nnls
a_nnls /= abc_sum
b_nnls /= abc_sum
c_nnls /= abc_sum

# Reconstruct
nnls_recon = a_nnls * ref_full + b_nnls * two_hex_modified_full + c_nnls * hex_full

r_nnls, p_nnls = pearsonr(Y_full, nnls_recon)
rms_nnls = compute_rms(Y_full, nnls_recon)

print("[FULL RANGE ANALYSIS - Single-pass NNLS]")
print(f"  a (1-hexanol):   {a_nnls*100:.2f}%")
print(f"  b (2-hexanol):   {b_nnls*100:.2f}%")
print(f"  c (hexane):      {c_nnls*100:.2f}%")
print(f"  Pearson r:       {r_nnls:.4f} (p-value={p_nnls:.4e})")
print(f"  RMS error:       {rms_nnls:.4f}")

plt.figure(figsize=(10,6))
plt.plot(wl_full, Y_full, label="Test Spectrum (Scaled)", color="red", linewidth=2)
plt.plot(wl_full, nnls_recon, label="NNLS Full-Range Fit", color="blue", linestyle="--", linewidth=2)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title("Full-Range NNLS Fit (1-hexanol, 2-hexanol*above2000->1hex, hexane)")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("FINAL-full_reconstruction_NNLS.png")
plt.show()

# FINISHED

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize
from scipy.stats import pearsonr

# ---------------------------------------------------
# 1) LOAD & PREPARE DATA
# ---------------------------------------------------
REF_FILE     = "reference.csv"      # 1‑hexanol reference
TEST_FILE    = "test.csv"           # Test sample
HEXANE_FILE  = "hexane.csv"         # Hexane
TWO_HEX_FILE = "2hexanol-update.csv"# 2‑hexanol
THF_FILE     = "thf.csv"            # THF (already 0–1, no scaling)

# Load CSVs
ref_df     = pd.read_csv(REF_FILE,     header=None, names=["wavelength", "transmittance"])
test_df    = pd.read_csv(TEST_FILE,    header=None, names=["wavelength", "transmittance"])
hexane_df  = pd.read_csv(HEXANE_FILE,  header=None, names=["wavelength", "transmittance"])
two_hex_df = pd.read_csv(TWO_HEX_FILE, header=None, names=["wavelength", "transmittance"])
thf_df     = pd.read_csv(THF_FILE,     header=None, names=["wavelength", "transmittance"])

# Convert test, 2-hex, hexane from 0–100% to 0–1 if not already
test_df["transmittance"]    /= 100.0
two_hex_df["transmittance"] /= 100.0
hexane_df["transmittance"]  /= 100.0

# Sort by wavelength
for df in [ref_df, test_df, hexane_df, two_hex_df, thf_df]:
    df.sort_values(by="wavelength", inplace=True)

# ---------------------------------------------------
# 2) SCALE TEST, HEXANE, & 2‑HEXANOL TO THE REFERENCE
#    but DO NOT scale THF
# ---------------------------------------------------
def scale_to_reference(df, ref_min, ref_max):
    orig_min = df["transmittance"].min()
    orig_max = df["transmittance"].max()
    df["transmittance_scaled"] = (
        (df["transmittance"] - orig_min)
        / (orig_max - orig_min)
    ) * (ref_max - ref_min) + ref_min
    return df

ref_min = ref_df["transmittance"].min() 
ref_max = ref_df["transmittance"].max()

test_df   = scale_to_reference(test_df,   ref_min, ref_max)
hexane_df = scale_to_reference(hexane_df, ref_min, ref_max)
two_hex_df= scale_to_reference(two_hex_df,ref_min, ref_max)
# THF remains unscaled (already 0–1)

# ---------------------------------------------------
# 3) DEFINE FULL OVERLAPPING RANGE
# ---------------------------------------------------
lower_bound = 500
upper_bound = 4000

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
# Clamp to user-defined range
full_min = max(full_min, lower_bound)
full_max = min(full_max, upper_bound)

common_wl = np.linspace(full_min, full_max, 1500)

# ---------------------------------------------------
# 4) INTERPOLATE
# ---------------------------------------------------
ref_interp         = np.interp(common_wl, ref_df["wavelength"],     ref_df["transmittance"])
test_interp        = np.interp(common_wl, test_df["wavelength"],    test_df["transmittance_scaled"])
hexane_interp      = np.interp(common_wl, hexane_df["wavelength"],  hexane_df["transmittance_scaled"])
two_hex_modified   = np.interp(common_wl, two_hex_df["wavelength"], two_hex_df["transmittance_scaled"])
thf_interp         = np.interp(common_wl, thf_df["wavelength"],     thf_df["transmittance"])

# ---------------------------------------------------
# 5) BUILD DESIGN MATRIX
#    ORDER: [1-hexanol, 2-hexanol_modified, hexane, THF]
# ---------------------------------------------------
X = np.column_stack([
    ref_interp,
    two_hex_modified,
    hexane_interp,
    thf_interp
])
Y = test_interp

# ---------------------------------------------------
# 6) SINGLE-PASS NNLS => 4 Fractions
# ---------------------------------------------------
coeffs_nnls, rnorm = nnls(X, Y)  # ensures all coefficients >= 0

# Normalize so sum=1
sum_nnls = np.sum(coeffs_nnls)
if sum_nnls > 0:
    coeffs_nnls /= sum_nnls

a_nnls = coeffs_nnls[0]
b_nnls = coeffs_nnls[1]
c_nnls = coeffs_nnls[2]
d_nnls = coeffs_nnls[3]

# ---------------------------------------------------
# 7) RECONSTRUCTION (NNLS) & EVALUATION
# ---------------------------------------------------
recon_nnls = a_nnls*ref_interp + b_nnls*two_hex_modified + c_nnls*hexane_interp + d_nnls*thf_interp

def compute_rms(u, v):
    return np.sqrt(np.mean((u - v)**2))

r_val_nnls, p_val_nnls = pearsonr(Y, recon_nnls)
rms_val_nnls = compute_rms(Y, recon_nnls)

print("\n[SINGLE-PASS NNLS: 4 COMPONENTS]")
print(f"  1-hexanol fraction:   {a_nnls*100:.2f}%")
print(f"  2-hexanol fraction:   {b_nnls*100:.2f}%")
print(f"  Hexane fraction:      {c_nnls*100:.2f}%")
print(f"  THF fraction:         {d_nnls*100:.2f}%")
print(f"  Pearson r:            {r_val_nnls:.4f} (p-value={p_val_nnls:.4e})")
print(f"  RMS error:            {rms_val_nnls:.4f}")


# ---------------------------------------------------
# 8) OPTIMIZE FOR PEARSON CORRELATION
#    - We maximize r => minimize -r
#    - Nonnegative & sum=1 constraints
# ---------------------------------------------------
def negative_pearson(coeffs, X, Y):
    # Reconstruct
    recon = np.dot(X, coeffs)
    # If all zeros or something degenerate, handle safely
    if np.all(recon == 0):
        return 1.0  # correlation = 0 => negative is 0 => cost is 1 is arbitrary
    r, _ = pearsonr(Y, recon)
    return -r  # minimize negative => maximize r

# Constraints: sum(coeffs) = 1
constraints = {'type': 'eq', 'fun': lambda c: np.sum(c) - 1}
# Bounds: all >= 0
bounds = [(0, None)] * X.shape[1]

# Start guess: uniform
x0 = np.ones(X.shape[1]) / X.shape[1]

result = minimize(
    fun=negative_pearson,
    x0=x0,
    args=(X, Y),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

coeffs_corr = result.x

# Ensure no negative from numerical slips
coeffs_corr[coeffs_corr < 0] = 0.0
# Renormalize if needed
s = coeffs_corr.sum()
if s > 0:
    coeffs_corr /= s

a_corr = coeffs_corr[0]
b_corr = coeffs_corr[1]
c_corr = coeffs_corr[2]
d_corr = coeffs_corr[3]

# Reconstruct & evaluate
recon_corr = a_corr*ref_interp + b_corr*two_hex_modified + c_corr*hexane_interp + d_corr*thf_interp
r_val_corr, p_val_corr = pearsonr(Y, recon_corr)
rms_val_corr = compute_rms(Y, recon_corr)

print("\n[PEARSON-CORRELATION OPTIMIZATION: 4 COMPONENTS]")
print(f"  1-hexanol fraction:   {a_corr*100:.2f}%")
print(f"  2-hexanol fraction:   {b_corr*100:.2f}%")
print(f"  Hexane fraction:      {c_corr*100:.2f}%")
print(f"  THF fraction:         {d_corr*100:.2f}%")
print(f"  Pearson r:            {r_val_corr:.4f} (p-value={p_val_corr:.4e})")
print(f"  RMS error:            {rms_val_corr:.4f}")


# ---------------------------------------------------
# 9) PLOT COMPARISONS
# ---------------------------------------------------
# -- (A) NNLS Fit --
plt.figure(figsize=(10,6))
plt.plot(common_wl, Y, label="Test Spectrum", color="red", linewidth=2)
plt.plot(common_wl, recon_nnls, label="Reconstructed (NNLS)", color="black", linestyle="--", linewidth=2)
plt.plot(common_wl, ref_interp, label="1-hexanol (Reference)", color="blue", linestyle="-.", linewidth=1)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title(f"NNLS Fit\n1-hex: {a_nnls*100:.2f}%, 2-hex: {b_nnls*100:.2f}%, Hexane: {c_nnls*100:.2f}%, THF: {d_nnls*100:.2f}%")
plt.gca().invert_xaxis()  # IR convention
plt.legend()
plt.tight_layout()
plt.savefig("nnls_fit.png")
plt.show()

# -- (B) Pearson-corr Fit --
plt.figure(figsize=(10,6))
plt.plot(common_wl, Y, label="Test Spectrum", linewidth=2)
plt.plot(common_wl, recon_corr, label="Reconstructed (Max Corr)", linestyle="--", linewidth=2)
plt.plot(common_wl, ref_interp, label="1-hexanol (Reference)", linestyle="-.", linewidth=1)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title(f"Correlation-Optimized Fit\n1-hex: {a_corr*100:.2f}%, 2-hex: {b_corr*100:.2f}%, Hexane: {c_corr*100:.2f}%, THF: {d_corr*100:.2f}%")
plt.gca().invert_xaxis()
plt.legend()
plt.tight_layout()
plt.savefig("corr_fit.png")
plt.show()


# ---------------------------------------------------
# 10) SHOW ALL SCALED SPECTRA IN A GRID
#     We'll just illustrate them individually.
# ---------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Because you have 5 key dataframes, you can fill 5 subplots; 
# the 6th subplot will be empty or you can use it for something else.
# For each, we'll plot "transmittance_scaled" if it exists, else "transmittance".

# (1) Reference
axes[0,0].plot(ref_df["wavelength"], ref_df["transmittance"])
axes[0,0].invert_xaxis()
axes[0,0].set_title("1‑Hexanol (Reference)")

# (2) Test (scaled)
axes[0,1].plot(test_df["wavelength"], test_df["transmittance_scaled"])
axes[0,1].invert_xaxis()
axes[0,1].set_title("Test (scaled)")

# (3) Hexane (scaled)
axes[0,2].plot(hexane_df["wavelength"], hexane_df["transmittance_scaled"])
axes[0,2].invert_xaxis()
axes[0,2].set_title("Hexane (scaled)")

# (4) 2-Hexanol (scaled)
axes[1,0].plot(two_hex_df["wavelength"], two_hex_df["transmittance_scaled"])
axes[1,0].invert_xaxis()
axes[1,0].set_title("2‑Hexanol (scaled)")

# (5) THF (not scaled)
axes[1,1].plot(thf_df["wavelength"], thf_df["transmittance"])
axes[1,1].invert_xaxis()
axes[1,1].set_title("THF (unscaled)")

# If you want to show the final 'test_interp' or the reconstruction in the 6th subplot:
axes[1,2].plot(common_wl, Y, label="Test Interp")
axes[1,2].plot(common_wl, recon_nnls, label="NNLS Recon", linestyle='--')
axes[1,2].plot(common_wl, recon_corr, label="Pearson Recon", linestyle='-.')
axes[1,2].invert_xaxis()
axes[1,2].set_title("Test vs NNLS Recon vs Pearson Recon")
axes[1,2].legend()

plt.tight_layout()
plt.savefig("all_spectra_grid.png")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.stats import pearsonr

# ---------------------------------------------------
# 1) LOAD & PREPARE DATA
# ---------------------------------------------------
REF_FILE     = "reference.csv"      # 1‑hexanol reference
TEST_FILE    = "test.csv"           # Test sample
HEXANE_FILE  = "hexane.csv"         # Hexane
TWO_HEX_FILE = "2hexanol.csv"       # 2‑hexanol
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

ref_min = ref_df["transmittance"].min()  # often ~0
ref_max = ref_df["transmittance"].max()  # e.g. ~0.91 or so

test_df   = scale_to_reference(test_df,   ref_min, ref_max)
hexane_df = scale_to_reference(hexane_df, ref_min, ref_max)
two_hex_df= scale_to_reference(two_hex_df,ref_min, ref_max)
# THF remains unscaled; we assume it's already 0–1

# ---------------------------------------------------
# 3) DEFINE FULL OVERLAPPING RANGE (e.g., 500–4000 cm⁻¹)
# ---------------------------------------------------
lower_bound = 500
upper_bound = 4000

# Find actual overlap among the five DataFrames
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

# Also clamp to the user-defined (500, 4000)
full_min = max(full_min, lower_bound)
full_max = min(full_max, upper_bound)

common_wl = np.linspace(full_min, full_max, 1500)  # 1500 points

# ---------------------------------------------------
# 4) INTERPOLATE & MODIFY 2‑HEXANOL ABOVE 2000 cm⁻¹
# ---------------------------------------------------
ref_interp     = np.interp(common_wl, ref_df["wavelength"],     ref_df["transmittance"])
test_interp    = np.interp(common_wl, test_df["wavelength"],    test_df["transmittance_scaled"])
hexane_interp  = np.interp(common_wl, hexane_df["wavelength"],  hexane_df["transmittance_scaled"])
two_hex_raw    = np.interp(common_wl, two_hex_df["wavelength"], two_hex_df["transmittance_scaled"])
thf_interp     = np.interp(common_wl, thf_df["wavelength"],     thf_df["transmittance"])

# Replace 2‑hex data above 2000 with reference (1‑hexanol)
two_hex_modified = np.where(common_wl > 2000, ref_interp, two_hex_raw)

# ---------------------------------------------------
# 5) BUILD DESIGN MATRIX FOR SINGLE-PASS NNLS
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
# 6) RUN NONNEGATIVE LEAST SQUARES => 4 Fractions
# ---------------------------------------------------
coeffs, rnorm = nnls(X, Y)  # ensures all >= 0

# Normalize so sum=1
sum_coeff = np.sum(coeffs)
if sum_coeff > 0:
    coeffs /= sum_coeff

a = coeffs[0]  # 1-hexanol fraction
b = coeffs[1]  # 2-hexanol fraction
c = coeffs[2]  # hexane fraction
d = coeffs[3]  # THF fraction

# ---------------------------------------------------
# 7) RECONSTRUCT & EVALUATE
# ---------------------------------------------------
recon = a*ref_interp + b*two_hex_modified + c*hexane_interp + d*thf_interp

# Pearson correlation & RMS
def compute_rms(u, v):
    return np.sqrt(np.mean((u - v)**2))

r_val, p_val = pearsonr(Y, recon)
rms_val = compute_rms(Y, recon)

print("\n[SINGLE-PASS NNLS: 4 COMPONENTS]")
print(f"  1-hexanol fraction:   {a*100:.2f}%")
print(f"  2-hexanol fraction:   {b*100:.2f}%")
print(f"  Hexane fraction:      {c*100:.2f}%")
print(f"  THF fraction:         {d*100:.2f}%")
print(f"  Pearson r:            {r_val:.4f} (p-value={p_val:.4e})")
print(f"  RMS error:            {rms_val:.4f}")

# ---------------------------------------------------
# 8) PLOT FINAL
# ---------------------------------------------------
plt.figure(figsize=(12,8))
plt.plot(common_wl, Y, label="Test Spectrum", color="red", linewidth=2)
plt.plot(common_wl, recon, label="Reconstructed", color="black", linestyle="--", linewidth=2)
plt.plot(common_wl, ref_interp, label="1-hexanol (Reference)", color="blue", linestyle="-.", linewidth=1)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance")
plt.title(f"Single-Pass NNLS Fit 1-hexanol: {a*100:.2f}%, 2-hexanol: {b*100:.2f}%, Hexane: {c*100:.2f}%, THF: {d*100:.2f}%")
plt.gca().invert_xaxis()  # IR convention
plt.legend()
plt.tight_layout()
plt.savefig("final_single_pass_4component.png")
plt.show()

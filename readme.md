# README: IR Spectral Fitting Analysis

This document explains the Python script used to perform spectral fitting for the identification and quantification of components in the IR spectrum of a synthesized product. The script analyzes experimental IR data by fitting it against known reference spectra of suspected components.

## Overview

This Python script:

1. Loads and preprocesses infrared (IR) spectroscopy data.
2. Scales and normalizes the data to a common reference.
3. Interpolates the data onto a shared wavelength scale.
4. Fits the experimental spectrum using two distinct methods:
   - Non-negative Least Squares (NNLS)
   - Pearson correlation optimization
5. Quantitatively analyzes and visualizes fitting results.

## Data

The analysis compares an experimental IR spectrum (`test_df`) against four known reference components:

- **1-Hexanol (target product)**
- **2-Hexanol** (potential side product)
- **Hexane** (residual solvent)
- **THF** (residual solvent)

Each component's spectrum is loaded from CSV files:

- `ref_df`: Reference spectrum of 1-hexanol
- `test_df`: Experimental product spectrum
- `two_hex_df`: 2-hexanol spectrum
- `hexane_df`: Hexane spectrum
- `thf_df`: Tetrahydrofuran spectrum

## Preprocessing

The spectra from test and solvents are scaled from percentage transmittance (0–100%) to a normalized scale of 0–1. THF is already normalized.

## Interpolation

To facilitate comparison, all spectra are interpolated onto a common wavelength axis (1500 data points), ensuring consistency in the fitting process.

## Spectral Fitting Approaches

Two methods are employed for spectral fitting:

### 1. Non-negative Least Squares (NNLS)

NNLS fitting provides a physically meaningful decomposition by constraining all component contributions to non-negative values. The method minimizes the squared residuals between the experimental and reconstructed spectra.

### 2. Pearson Correlation Optimization

The second fitting approach maximizes the Pearson correlation between the reconstructed and experimental spectra, emphasizing the overall spectral shape rather than just intensity matches.

## Results & Interpretation

- **NNLS Method:** Best amplitude match, indicating realistic residual solvent presence. Suggests:

  - 1-hexanol: \~37%
  - 2-hexanol: \~12%
  - Hexane: \~33%
  - THF: \~18%

- **Pearson Method:** Optimized shape similarity, slightly less realistic due to omission of plausible impurities. Suggests:

  - 1-hexanol: \~75%
  - THF: \~25%


## Visualizations

Generated plots clearly illustrate spectral comparisons and fitting quality:

- `nnls_fit.png`: Spectrum reconstructed using NNLS
- `corr_fit.png`: Spectrum optimized for Pearson correlation

## Usage

- Python script (`main2.py`) handles data preparation, fitting, and visualization.
- Dependencies: pandas, numpy, matplotlib, scipy




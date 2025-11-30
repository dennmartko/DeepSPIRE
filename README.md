# DeepSPIRE.

**TensorFlow Implementation**

Paper Link will be updated upon publication.

This repository contains the code and data to reproduce the results of the paper:
**"Super-resolving Herschel - a deep learning based deconvolution
and denoising technique I."** by Koopmans et al 2025, published in [TBD].

## 📌 Abstract
Aims. Dusty star-forming galaxies (DSFG) dominate the far-infrared (FIR) and sub-millimetre (sub-mm) number counts, but single-
dish surveys at these wavelengths suffer from poor angular resolution, making identifications of multi-wavelength counterparts diffi-
cult. Prior-driven deblending techniques require extensive fine-tuning and struggle to process large fields. This work aims to develop
a fast and reliable deep-learning based deconvolution and denoising super-resolution (SR) technique.
Methods. We employ a transformer neural network to improve the resolution of the Herschel/SPIRE 500 μm observations by a
factor of 4.5, with input comprised of Spitzer/MIPS 24μm and Herschel/SPIRE 250, 350, 500μm images. The network was trained on
simulations from SIDES and SHARK. To mimic realistic observations, we injected instrumental noise into the input simulated images,
while keeping the target images noise-free to enhance the de-noising capabilities of our method. We evaluated the performance of our
method on simulated test sets and real JCMT/SCUBA-2 450 μm observations in the COSMOS field which have superior resolution
compared to Herschel.
Results. Our SR method achieves an inference time of ∼ 1s/deg2 on consumer-grade GPUs, much faster than traditional deblending
techniques. Using the simulation test sets, we show that fluxes of the extracted sources from the super-resolved image are accurate
to within 5% for sources with an intrinsic flux ≳ 8 mJy, which is a substantial improvement compared to blind extraction on the
native images. Astrometric error is low, at 1′′compared to the 12′′pixel scale. In terms of reliability and completeness, ≳ 90% of
the extracted sources brighter than ∼ 3 mJy are reliable and more than 90% of the input sources with intrinsic fluxes ≳ 6 mJy are
recovered. When applied to the real 500 μm observations, the fluxes of the extracted sources from the super-resolved map agree well
with the SCUBA-2 measured fluxes (after converting the 450 μm fluxes to 500 μm using a correction factor of 0.84) for sources above
∼ 10 mJy. Our 500 μm number counts are also consistent with previous SCUBA-2 measurements. Thanks to its speed, our technique
enables SR over hundreds of deg2 without the need for fine-tuning, facilitating statistical analysis of DSFGs.

---


## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/dennmartko/DeepSPIRE.git
cd DeepSPIRE
```

## 2. Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

Main requirements:
- Python 3.8-3.12

## 3. Install dependencies & DeepSPIRE
```bash
pip install -r requirements.txt
pip install -e .
```


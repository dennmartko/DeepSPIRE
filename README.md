# DeepSPIRE

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
- Python 3.10-3.12
- pySIDES (for data preparation)
- TensorFlow 2.2.x (GPU version required)
- For Training: NVIDIA GPU with at least 40GB VRAM
- For Inference: NVIDIA GPU with at least 8GB VRAM

DeepSPIRE uses TensorFlow with GPU acceleration. To ensure proper GPU support, you need a compatible NVIDIA GPU, CUDA Toolkit, and cuDNN installed on your system. For more details on installing CUDA and cuDNN, see the official NVIDIA guides and TensorFlow GPU setup.

## 3. Install dependencies & DeepSPIRE
```bash
pip install -r requirements.txt
pip install -e .
```

## ♻️ Reproducing results

### 1. 📂 Data Preparation
1. Download the pySIDES Uchuu dataset from https://data.lam.fr/sides/search/dataset. We used all catalogs upto and including tile_6_8.
2. Use pySIDES to generate simulated catalogs containing Herschel 250, 350 and 500 μm and Spitzer MIPS 24 μm fluxes as well as the source coordinates. Follow the instructions in the pySIDES documentation: https://gitlab.lam.fr/mbethermin/sides-public-release.
3. Download the SHARK lightcone catalogs (Private communication with A. Lagos). One can also train with only the Uchuu dataset from pySIDES.
4. Run the script ``process_simulated_catalogs.py`` in the ``scripts/preprocess/`` folder to process the SIDES and/or SHARK catalogs into 2 deg² subcatalogs including only the necessary columns. Modify the output directory in the script as needed. This script will create processed catalogs in the specified output directory with an iterator for the suffix, i.e. SHARK_1.

Note: This is one of the few functions where some paths are still hardcoded. The filepath in load_catalog() function and the paths for the SHARK catalogs in load_shark_catalog() function need to be set according to your directory structure. output_dir in the main function also needs to be set.

5. Now, one has a directory with processed simulated catalogs, named SHARK_1, SHARK_2, ..., SIDES_1, SIDES_2, ... etc. Each 2 deg². Next, generate the 2 deg² datamaps using the mapmaker in pySIDES. There are two ways to do this: (1) all manuallyFollow the instructions in the pySIDES documentation or (2) using the provided notebook ``gen_sim_maps.ipynb`` in the ``preprocess/`` folder. The notebook uses pySIDES to generate the datamaps for all processed catalogs in a specified input directory and saves them to a specified output directory. Modify the input and output directories in the notebook as needed as well as the path to the pySIDES files. Keep the pixel sizes the same as in the paper. This script takes less than an hour to generate all the datamaps: the input maps and the target/SR maps.

Note: By default, pySIDES generates maps with perfect Gaussian beams. In our paper, we use a realistic 24 μm beam which comes from the EGG (Empirical Galaxy Generator) code. To use this beam, one needs to modify the `make_maps.py` script in the pySIDES code to load the beam ``data/beams/spitzer-mips24.fits`` instead and convolve with that beam when generating the 24 μm maps. We also provide an improved version of `make_maps.py` in the ``dev/`` folder which can be used instead (change the path to the beam file accordingly). 

Note2: The datamaps should have a naming convention as follows:
- Input maps: `SIDES_i_BAND_smoothed_Jy_beam.fits`
- Target maps: `SIDES_i_SR_SPIRE500_smoothed_Jy_beam.fits`
where `i` is the catalog number and `BAND` is one of `MIPS24`, `SPIRE250`, `SPIRE350`, or `SPIRE500`.

6. We are now ready to generate the training, validation and test datasets. Run the script ``gen_sim_data.py`` in the ``scripts/preprocess/`` folder. Modify the Config class in the script to set the input and target map directories, output dataset directory, number of samples, patch size, batch size, noise levels etc. as needed. This script can take up to a day to run. The script will create a directory structure as follows:
```
output_dataset_dir/
    SIDES_1/
        Train/
            24/24_0.fits
            24/24_1.fits
            ...
            250/
            350/
            500/
            500SR/
            500SR_mask/
        Validation/
            ...
        Test/
            ...
    etc.
```

7. Run the final script ``DataMerge.py`` in the ``scripts/preprocess/`` folder to merge all the individual dataset directories (SIDES_1, SIDES_2, ...) into a single `Train`, `Validation` and `Test` directory. Modify the `base` variable in the script to point to the directory `output_dataset_dir` created in the previous step.

### 2.  🧠 Training the DeepSPIRE model from the paper
Our training results are located in the `results/SwinUnet/deepSPIRE_default` folder. 

1. Modify and check the training configuration file located at ``configs/train/SwinUnet/TrainConfig.yaml`` as needed. The default configuration is the one used in the paper.
The data path pointing towards the dataset needs to point to ``output_dataset_dir``. Use a different `run_name` as otherwise it will resume training from the previous run.

2. Run the training script in the ``scripts/train/`` folder:
```bash
python train.py --config configs/train/SwinUnet/TrainConfig.yaml
```
This will start the training and create a new results folder in ``results/SwinUnet/{run_name}`` where the model checkpoints, logs and training history will be saved. Training can always be resumed by using the same `run_name` in the config file. It will automatically load the latest checkpoint and continue training. It will stop once the early stopping criteria is met or the maximum number of epochs is reached. Training the model with the default configuration requires a GPU with at least 40GB VRAM and 60-100 GB of RAM. Logs are time-stamped. During training, the script will make periodic plots of the training and validation loss curves as well as prediction examples on the validation set. These will be saved in the results folder.

### 3.  📊 Evaluating the DeepSPIRE model simulation results

The following steps show how to evaluate the trained DeepSPIRE model on the simulated test set to reproduce the results from the paper.

Evaluation requires a truth catalog of the test set, a SR catalog extracted from the super-resolved test images and an ideal target catalog extracted from the noise-free target test images.

1. Evalua

### 4.  📊 Application of the DeepSPIRE model on the COSMOS field 

<!-- Alternative symbols you can use: ♻️  🔁  🔬  ✅  🧪

- Recommended symbols for training (concise meanings)
    - 🧠 — model / architecture / training procedure
    - 🧪 — experiments / training runs / ablations
    - 🔁 — epochs / iterations / data augmentation loops
    - ⚙️ — configuration / hyperparameters / setup
    - ⏱️ — runtime / performance / speed
    - 💾 — checkpoints / saving / weights
    - ✅ — successful runs / completed steps / validation passed
    - 📊 — metrics / evaluation / plots
    - 🚀 — inference / deployment / production

- Short usage examples
    - "### 2. Training 🧠"
    - "#### Training runs 🧪"
    - "Configuration ⚙️ — hyperparams, batch size, optimizer"
    - "Checkpoints 💾 — save every N epochs"
    - "Results 📊 / ✅"

- Tips
    - Keep symbols consistent across headings and lists.
    - Use 1–2 symbols per section to avoid visual clutter.
    - Prefer semantic symbols (e.g., 🧪 for experiments) so readers scan quickly. -->


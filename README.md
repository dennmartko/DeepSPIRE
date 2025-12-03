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

Main requirements [ToDo: Check]:
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

To reproduce the results we provide the data needed that allows one to generate their own datasets, apply our trained model, and evaluate the performance using our evaluation scripts. These evaluation scripts will produce plots similar to those shown in the paper. However, due to random seeding results may slightly differ. We tried our best to fix the seeds and describe the environment in which we have performed our work. 

In the sections below, we will highlight all the steps we have taken to come to our main results and conclusions in the paper. We will highlight where the provided data can be used instead which skips a lot of unnecessary steps when trying to reproduce our results. Nevertheless, please read all the steps to gain a good understanding of the workflow. With the provided data, one can start at step (6) in the Data Preparation section below. 

### 1. 📂 Data Preparation
1. Download the pySIDES Uchuu dataset from https://data.lam.fr/sides/search/dataset. We used all catalogs upto and including tile_6_8.
2. Use pySIDES to generate simulated catalogs containing Herschel 250, 350 and 500 μm and Spitzer MIPS 24 μm fluxes as well as the source coordinates. Follow the instructions in the pySIDES documentation: https://gitlab.lam.fr/mbethermin/sides-public-release.
3. Download the SHARK lightcone catalogs (Private communication with Claudia Lagos). One can also train with only the Uchuu dataset from pySIDES, although results may differ more.
4. Run the script ``process_simulated_catalogs.py`` in the ``scripts/preprocess/`` folder to process the SIDES and/or SHARK catalogs into 2 deg² subcatalogs including only the necessary columns. Modify the output directory in the script as needed. This script will create processed catalogs in the specified output directory with an iterator for the suffix, i.e. SHARK_1.

Note: This is one of the few functions where some paths are still hardcoded. The filepath in load_catalog() function and the paths for the SHARK catalogs in load_shark_catalog() function need to be set according to your directory structure. output_dir in the main function also needs to be set.

5. Now, one has a directory with processed simulated catalogs, named SHARK_1, SHARK_2, ..., SIDES_1, SIDES_2, ... etc. Each 2 deg². Next, generate the 2 deg² datamaps using the mapmaker in pySIDES. There are two ways to do this: (1) all manuallyFollow the instructions in the pySIDES documentation or (2) using the provided notebook ``gen_sim_maps.ipynb`` in the ``preprocess/`` folder. The notebook uses pySIDES to generate the datamaps for all processed catalogs in a specified input directory and saves them to a specified output directory. Modify the input and output directories in the notebook as needed as well as the path to the pySIDES files. Keep the pixel sizes the same as in the paper. This script takes less than an hour to generate all the datamaps: the input maps and the target/SR maps.

Note: By default, pySIDES generates maps with perfect Gaussian beams. In our paper, we use a realistic 24 μm beam which comes from the EGG (Empirical Galaxy Generator) code. To use this beam, one needs to modify the `make_maps.py` script in the pySIDES code to load the beam ``data/beams/spitzer-mips24.fits`` instead and convolve with that beam when generating the 24 μm maps. We also provide an improved version of `make_maps.py` in the ``dev/`` folder which can be used instead (change the path to the beam file accordingly). 

Note2: The datamaps should have a naming convention as follows:
- Input maps: `SIDES_i_BAND_smoothed_Jy_beam.fits`
- Target maps: `SIDES_i_SR_SPIRE500_smoothed_Jy_beam.fits`
where `i` is the catalog number and `BAND` is one of `MIPS24`, `SPIRE250`, `SPIRE350`, or `SPIRE500`.

6. You are now ready to generate the training, validation and test dataset. Download the provided data and place the subdirectories into ``data/`` as needed. Then run ``gen_sim_data.py`` in the ``scripts/preprocess/`` folder. Update the `Config` class in the script to specify the input and target map directories, output dataset directory, and noise levels. To reproduce our results, only adjust the paths. The default dataset names and paths are used throughout our scripts. This script may require up to one day to complete and generally runs much faster on consumer CPU's, benifitting the higher clock speeds as MPI is not optimally handled by 3rd party packages.

This script will produce the following directory structure:
```text
output_dataset_dir/
├── SIDES_1/
│   ├── Train/
│   │   ├── 24/
│   │   │   └── 24_0.fits ... 24_N.fits
│   │   ├── 250/
│   │   │   └── 250_0.fits ... 250_N.fits
│   │   ├── 350/
│   │   │   └── 350_0.fits ... 350_N.fits
│   │   ├── 500/
│   │   │   └── 500_0.fits ... 500_N.fits
│   │   ├── 500SR/
│   │   │   └── 500SR_0.fits ... 500SR_N.fits
│   │   └── 500SR_mask/
│   │       └── 500SR_mask_0.fits ... 500SR_mask_N.fits
│   ├── Validation/
│   │   └── ...
│   └── Test/
│       └── ...
├── ...
├── SIDES_30/
├── ...
└── SHARK_30/
```
Here, N represents the number of files in each directory. Another, similar, folder will be created which contains figures that can be used to analyse each individual SHARK or SIDES dataset to verify data quality and that the pipeline is working.

7. Run the final script ``DataMerge.py`` in the ``scripts/preprocess/`` folder to merge all the individual dataset directories (SIDES_1, SIDES_2, ...) into a single `Train`, `Validation` and `Test` directory. Ensure the `base` variable in the script points to the directory ``output_dataset_dir`` created in the previous step. This step is generally fast (~few minutes) when executed on SSD's / NVMe's.

One will now have a dataset file structure looking like this:
```text
output_dataset_dir/
├── Train/
│   ├── 24/
│   │   └── 24_0.fits ... 24_M.fits
│   ├── 250/
│   │   └── 250_0.fits ... 250_M.fits
│   ├── 350/
│   │   └── 350_0.fits ... 350_M.fits
│   ├── 500/
│   │   └── 500_0.fits ... 500_M.fits
│   ├── 500SR/
│   │   └── 500SR_0.fits ... 500SR_M.fits
│   └── 500SR_mask/
│       └── 500SR_mask_0.fits ... 500SR_mask_M.fits
├── Validation/
│   └── ...
└── Test/
    └── ...
```

### 2.  🧠 Training the DeepSPIRE model from the paper
Use the previously generated dataset located at ``output_dataset_dir`` to train the DeepSPIRE model. Use our provided training configuration to reproduce the training from the paper. Given that training introduces randomness, results may slightly differ. To reproduce our results from the paper as much as possible, you can use our trained model located in ``results/SwinUnet/deepSPIRE_default`` and skip this training section.

1. Modify and check the training configuration file located at ``configs/train/SwinUnet/config_deepSPIRE.yaml`` as needed. The default configuration is the one used in the paper.
The data path pointing towards the dataset needs to point to ``output_dataset_dir``. Use a different `run_name` as otherwise it will resume training using our trained model.

2. Run the training script in the ``scripts/train/`` folder:
```bash
python3 scripts/train/train.py --config configs/train/SwinUnet/config_deepSPIRE.yaml
```
This will start the training and create a new results folder in ``results/SwinUnet/{run_name}`` where the model checkpoints, logs and training history will be saved. Training can always be resumed by using the same `run_name` in the config file. It will automatically load the latest checkpoint and continue training. It will stop once the early stopping criteria is met or the maximum number of epochs is reached. Training the model with the default configuration requires a GPU with at least 40GB VRAM and 60-100 GB of RAM. Logs are time-stamped. During training, the script will refresh a plot of the training and validation loss curves as well as provide prediction examples on the validation set. These will be saved in the results folder.

### 3.  📊 Evaluating the DeepSPIRE model simulation results

The following steps show how to evaluate the trained DeepSPIRE model on the simulated test set. To reproduce the results from the paper, please use our trained model located in ``results/SwinUnet/deepSPIRE_default``, the provided catalog data located in ``data/simulation_data/catalogs`` as well as the dataset you have generated during the preprocessing. 

Before we can evaluate the model, we need to create the input catalog of the test set, extract sources from the native 500 μm test images as well as extract sources from both the target and super-resolved test images. All the following steps will have to be performed, regardless of whether you trained your own model or are using our trained model and using our provided data. You should be able to run all the steps on a consumer-grade GPU with at least 8 GB VRAM for the inference step.

1. First, create the input catalog of the test set by running the script ``scripts/evaluate/get_input_test_catalog.py``. Modify the hard-coded paths in the script as needed. This will create the input test catalog ``shark_sides_noisy_test_input_catalog.csv``(.CSV) in the specified output directory which only contains the `SMIPS24` an `SSPIRE500` flux columns. In case you want different columns, adapt the parameters. 

Note, that this script uses flux cuts to remove extremely faint sources from the catalog in order to lower file size. The default flux cuts do not affect the results in the paper but can be modified if needed (i.e. set to 0). The script uses parallel processing to read each SHARK/SIDES subcatalog and automatically only keeps sources covered in the test cutouts. This speeds up the catalog creation significantly and each process appends the input fluxes to the input test catalog in a safe manner. Consequently, never change the extension of the input catalog file as fits files can not be appended to. Our advice is to start with a small number of processes (the SIDES catalogs are ~1GB each). The final input test catalog will have a size slightly below 1GB and the entire process may take upto 20-30 minutes.

2. Next, extract sources from the native 500 μm test images by running the script ``scripts/evaluate/get_native_source_extracted_catalog.py``. Modify the hard-coded paths in the script as needed. This script will extract sources from the native (with noise) simulated 500 μm datamaps using `photutils` and only store the sources covered by the test images. The extracted catalog ``SPIRE500_native_catalog.fits`` (.FITS) will be saved in the specified output directory. The specifics on the source extraction methodology is written in the paper or can be inferred from the code. Here, staying away from sources detected near the image border was not that important for the purpose of the paper and sources can be extracted from anywhere. The size of the native catalog is typically around 1 MB.

3. The final two catalogs, the target and SR test catalog (.FITS) can be created by running (from the root directory):
```bash
python3 scripts/evaluate/get_SR_target_catalog.py --config configs/SwinUnet/evaluate/get_SR_target_catalog.yaml
```
This script requires a GPU with atleast a VRAM of 8 GB. Lower the inference batch size if OOM erors show up. This does not affect model performance. Modify only the config file for reproduction. The script will load the chosen trained DeepSPIRE model and run inference on the simulated test set to super-resolve the test images. It will extract sources from both the target and super-resolved images using a border padding of 8 pixels (variable `border_pad` in the script) as stated in the paper to avoid border effects. The target catalog ``500SR_target_catalog.fits`` is stored in the specified output directory. The SR catalog is always stored in the results folder of the chosen trained model in ``/results/SwinUnet/{run_name}/testing/simulation_results/500SR_SR_catalog.fits``. The size of both catalogs is typically around 5 MB each.

This script heavily relies on parallel processing to speed up source extraction. The number of CPU cores directly scales with memory consumption. However, for most systems, using all available CPU cores should work fine.

4. Finally, we evaluate our model by running (from the root directory):
```bash
python3 scripts/evaluate/evaluate_simulations.py --config configs/SwinUnet/evaluate/evaluate_simulations.yaml
```
This script will generate all the plots shown in the paper for the simulation results including flux reproduction, astrometric accuracy, reliability and completeness as well as the image-based comparisons. Modify the config file as needed to point to the correct catalog paths. The plots will be saved in the results folder of the chosen trained model in ``/results/SwinUnet/{run_name}/testing/simulation_results/``. In order to reproduce the exact results from the paper, do not modify the evaluation script. This script requires a GPU with atleast a VRAM of 8 GB for the image comparison plots. Figures get overwritten if the script is run multiple times with the same config. We have provided the plots from the paper already in the results folder.

### 4.  🔭 Application of the DeepSPIRE model on the COSMOS field & Evaluation with SCUBA-2

The following steps show how to apply the trained DeepSPIRE model on the real Spitzer MIPS 24 μm & Herschel SPIRE observations of the COSMOS field and reproduce the results from the paper. This requires the real COSMOS maps as well as SCUBA-2 450 μm imaging + catalog for validation. Moreover, we have to apply background subtraction on the real SPIRE maps before creating a dataset consisting of cutouts. These background maps are computed from the estimated background fits in the XID+ catalogs.

We provide the observational data maps and catalogs used in the paper in the ``data/observation_data/`` folder from the downloadable link. Moreover, we provide the corrected COSMOS datamaps in the ``data/observation_data/corrected_maps/`` folder. Therefore, one can skip steps (1) to (3) below if using our provided data.

1. Download the Herschel SPIRE 250, 350, 500 μm maps of the COSMOS field from ``https://hedam.lam.fr/HELP/dataproducts/dmu19/dmu19_HerMES/data/``. Select the COSMOS-NEST_image_250/350/500 SMAP fits files. Download the Spitzer MIPS 24 μm map of the COSMOS field from ``https://irsa.ipac.caltech.edu/data/COSMOS/images/spitzer/mips/mips_24_GO3_sci_10.fits``. The SPIRE maps are in units of Jy/beam while the MIPS map is in units of MJy/sr.

2. During paper writing, the SCUBA-2 450 μm COSMOS map and catalog were obtained from private communication with the STUDIES team. The paper is now published: ``https://ui.adsabs.harvard.edu/abs/2024ApJ...971..117G/abstract`` with information on where to download the data (https://group.asiaa.sinica.edu.tw/whwang/studies/cosmos_final/). The SCUBA-2 450 μm map (SC_450.fits) is in units of mJy/beam and catalog (STUDIES-COSMOS_450um_v20230206.fits) in units of mJy.

3. Next, comes preprocessing the real COSMOS maps. Here, we need to correct for unit conversions, side-lobe corrections and background subtraction. To correct the SPIRE and MIPS maps for background, we use the background estimates from the HELP XID+ COSMOS catalog which can be downloaded from ``https://hedam.lam.fr/HELP/dataproducts/dmu26/dmu26_XID+COSMOS2024/Master_Post_Catalogue_MIPS_PACS_SPIRE_SCUBA.fits``. By default, the path of this catalog is expected to be ``data/observation_data/Master_Post_Catalogue_MIPS_PACS_SPIRE_SCUBA.fits``. Note, the file size is 15GB. 

Run the notebook ``scripts/preprocess/correct_obs_maps.ipynb``. Modify the hard-coded input and output paths in the notebook as needed. The code expects all observational datamaps to be within a single directory path, e.g. `data/observation_data`. The notebook will create different versions of the corrected datamaps using various methods to estimate the background in each pixel. We will use the datamaps with suffix ``interp_bkg_subtracted`` for the rest of the analysis as these were used in the paper. 
The notebook may take a while to run and ensure enough RAM to load in the XID+ catalog. 

4. We are now ready to create the observational COSMOS dataset. Run the script ``scripts/preprocess/gen_obs_data.py``. Modify the `Config` class in the script to set the input and output directories, classes etc as needed. This script requires that the bounds in sky coordinates of the COSMOS field are pre-defined. We have two sets of bounds hardcoded in the script: one for the smaller area covered by SCUBA-2 and one for the full 2.2 deg² COSMOS field as defined by the limiting Spitzer MIPS 24 μm coverage. Make sure to comment/uncomment the correct set of bounds in the script as needed. The script will produce similar outputs as the simulated dataset generation script. However, there will only be a `Test` set and no merging is needed.

To reproduce the results from the paper, generate two datasets (separate output directories), one for the full COSMOS field and one for the SCUBA-2 covered area. Use the corresponding bounds and for SCUBA-2 add another entry in the config variable lists by adding the class name 450, designating it as target and using 0 for hdu_idx. For the super-resolution of the entire COSMOS field, we only generate a dataset containing input classes (24, 250, 350 and 500). Give the two datasets different names. The script may take upto an hour to complete.


5. First, we super-resolve the COSMOS Herschel SPIRE 500 μm data covered by the smaller SCUBA-2 footprint. Run (from the root directory):
```bash
python3 scripts/evaluate/evaluate_observations.py --config configs/SwinUnet/evaluate/evaluate_observations.yaml
```

Ensure that the config file contains the 450 class designated as the target class. Modify the config file as needed to point to the correct dataset directory and the correct model/run name. The script will create the plots (flux reproduction, image comparison) from the paper and stores them in the results folder of the chosen trained model in ``/results/SwinUnet/{run_name}/testing/observation_results/``. 

6. Second, we super-resolve the full COSMOS Herschel SPIRE 500 μm data. We will only create the super-resolved catalog extracted from the super-resolved images. Run (from the root directory):
```bash
python3 scripts/evaluate/get_SR_catalog_observations.py --config configs/SwinUnet/evaluate/get_SR_target_catalog_obs.yaml
```

This script requires a GPU with atleast a VRAM of 8 GB. Again, lower the inference batch size if OOM erors occur. Do not modify this script, but only the config file. The script will load the chosen trained deepSPIRE model and super-resolve the full COSMOS dataset. Finally, it will extract sources from the super-resolved images. The SR catalog is always stored in the results folder of the chosen trained model: ``/results/SwinUnet/{run_name}/testing/observation_results/{dataset_dir}_SR_catalog.fits`` where `datset_dir` is the name of the chosen dataset directory in step (4). The size of the SR catalog is typically <1 MB.

Note, we do not ignore sources extracted near the image border here as `border_pad = 0`.

7. Using ``/results/SwinUnet/{run_name}/testing/observation_results/{dataset_dir}_SR_catalog.fits``, one can now create the (un)corrected 500 μm number counts as shown in the paper. Modify (use the COSMOS SR catalog) and run the notebook ``scripts/evaluate/number_counts.ipynb``. The notebook will create the (un)corrected number counts plots from the paper. The plots will be saved in the same folder as the script.

Note: This notebook requires the XID+ COSMOS catalog as it is used in the comparison plots. Make sure you have downloaded this catalog as mentioned in step (3) above and placed it in the correct directory.
Note2: To calculate the correct number counts, you need to make another test input catalog (from simulations) but without any restrictions on extracting sources near the image border, i.e. `border_pad = 0`.

🚀 You have now reproduced all the results from the paper! 🚀
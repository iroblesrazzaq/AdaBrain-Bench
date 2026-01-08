# TODO:

Later:
- Figure out SHHS download - have ~100 files left, keep getting api blocked
- Preprocess all

- skipping SHHS for now

Requested Permission, download later:
- SEED, SEED-IV, SEED-VIG

Finish Downloading:
- SHHS (annoying)


Downloaded + Formatted(✅):
- EEGMAT ✅
- BCI-IV-2A ✅
- Things-EEG ✅
- Siena ✅
- HMC ✅
- Sleep-EDF ✅
- SHU ✅
- TUAB ✅
- TUEV ✅


Preprocessed:
- EEGMAT ✅
- BCI-IV-2A ✅
- HMC ✅
- SHU ✅
- TUEV ✅
- Siena ✅
- Sleep-EDF 
- Things-EEG 
- TUAB 


Splits created:
- EEGMAT ✅
- BCI-IV-2A ✅
- HMC ✅
- SHU ✅
- Siena ✅





# Improvements:
- bug fixes: 
    - BCI-4-2A -> BCI-IV-2A dataset naming consistency
    - standardized preprocessed directory and file naming structure to fix errors for Siena, TUEV, and SHU datasets
- 56% average speedup in json train/test/val split creation script processing
- memory safe file loading

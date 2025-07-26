# CausalMER

## Official Pytorch Implementation of [Causal Inference for Modality Debiasing in Multimodal Emotion Recognition](https://www.mdpi.com/2076-3417/14/23/11397)
#### Authors: [Juyeon Kim†](https://scholar.google.com/citations?user=5xo177UAAAAJ&hl=ko&oi=sra), [Juyoung Hong†](https://scholar.google.com/citations?user=fMNgQtMAAAAJ&hl=ko&oi=sra), [Yukyung Choi](https://scholar.google.com/citations?user=vMrPtrAAAAAJ&hl=ko&oi=sra)
† These authors contributed equally to this work.

![2024-12-24 142641](https://www.mdpi.com/applsci/applsci-14-11397/article_deploy/html/images/applsci-14-11397-g004.png)


## Abstract
 Multimodal emotion recognition (MER) aims to enhance the understanding of human emotions by integrating visual, auditory, and textual modalities. However, previous MER approaches often depend on a dominant modality rather than considering all modalities, leading to poor generalization. To address this, we propose Causal Inference in Multimodal Emotion Recognition (CausalMER), which leverages counterfactual reasoning and causal graphs to capture relationships between modalities and reduce direct modality effects contributing to bias. This allows CausalMER to make unbiased predictions while being easily applied to existing MER methods in a model-agnostic manner, without requiring any architectural modifications. We evaluate CausalMER on the IEMOCAP and CMU-MOSEI datasets, widely used benchmarks in MER, and compare it with existing methods. On the IEMOCAP dataset with the MulT backbone, CausalMER achieves an average accuracy of 83.4%. On the CMU-MOSEI dataset, the average accuracies with MulT, PMR, and DMD backbones are 50.1%, 48.8%, and 48.8%, respectively. Experimental results demonstrate that CausalMER is robust in missing modality scenarios, as shown by its low standard deviation in performance drop gaps. Additionally, we evaluate modality contributions and show that CausalMER achieves balanced contributions from each modality, effectively mitigating direct biases from individual modalities.
 
> **PDF**: [Causal Inference for Modality Debiasing in Multimodal Emotion Recognition](https://www.mdpi.com/2076-3417/14/23/11397/pdf?version=1733500973)

---

## Usage

## Prerequisites

### Recommended Environment
* We strongly recommend following the environment, which is very important as to whether it's reproduced or not.
    * OS : Ubuntu 18.04
    * CUDA : 11.7
    * Python 3.8
    * Pytorch 1.13.0 Torchvision 0.14.0
    * GPU : NVIDA A6000 (48G)
* Required packages are listed in **environments.yaml**. You can install by running:
```
conda env create -f environments.yaml
conda activate causalmer
```
   
### Data Preparation

This project utilizes pre-processed versions of the IEMOCAP and CMU-MOSEI datasets, as provided by the MulT.

To replicate the experiments, please download the specific pre-processed dataset from the official MulT repository:

- **Download Link**: [MulT GitHub Repository](https://github.com/yaohungt/Multimodal-Transformer)

Please follow the instructions provided in the MulT repository to set up the data.

#### Original Data Sources
The original datasets can be found at:
- **IEMOCAP**: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)
- **CMU-MOSEI**: [https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

* The directory structure after downloading data should be as follows.

~~~~
├── CausalMER
   ├── data
      ├── iemocap_data.pkl
      ├── iemocap_test_a.dt
      ├── iemocap_test_na.dt
      ├── ...
      └── mosi_valid_na.dt
~~~~

* Please note that the parameters provided are the initial parameters before any training has been conducted.
   * ckpt : [Google Drive](https://drive.google.com/file/d/1u9iLjwW0rbXJIgyoePLMOp-Py60WCQOG/view?usp=sharing)

* The **checkpoint file**  should be organized as follows:
~~~~
├──  CausalMER
   ├── checkpoint(iemocap)
      ├── checkpoint_iemocap_MULT.pt
~~~~

## Run

### Training
```
# bash main.sh {gpu_number} {ckpt_name} {seed} {dataset_name}
bash main.sh 7 causalmer_iemocap 0 iemocap
```

### Inference
```
# bash inference.sh {gpu_number} {ckpt_name} {seed} {dataset_name}
bash inference.sh 5 causalmer_iemocap 0 iemocap
```

## References
We referenced the repos below for the code. Thank you!!
* [Mult](https://github.com/yaohungt/Multimodal-Transformer?tab=readme-ov-file)
* [CF-VQA](https://github.com/yuleiniu/cfvqa)

---

## Citation
If our work is useful in your research, please consider citing our paper:
```
@article{kim2024causal,
  title={Causal Inference for Modality Debiasing in Multimodal Emotion Recognition},
  author={Kim, Juyeon and Hong, Juyoung and Choi, Yukyung},
  journal={Applied Sciences},
  volume={14},
  number={23},
  pages={11397},
  year={2024},
  publisher={MDPI}
}
```


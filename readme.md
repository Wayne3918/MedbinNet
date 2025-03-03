# MedBin: A Lightweight End-to-End Model-based Method for Medical Waste Management

This repository contains the implementation of **MedBinNet**, a model designed for efficient medical waste management. Our approach is based on MMDetection, and the detailed instructions below outline the prerequisites, data preparation, and steps to train and test the model.

---

## Table of Contents

- [Overview](#overview)
- [Publication](#publication)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Configurations](#configurations)
- [Citation](#citation)
- [License](#license)

---

## Overview

**MedBinNet** is a lightweight, end-to-end model designed to assist with the detection and management of medical waste. Our approach leverages state-of-the-art detection techniques from MMDetection, providing a robust framework for both academic research and practical applications.

---

## Publication

For more details on the underlying research and methodology, please refer to our publication in *Waste Management*:

[MedBin: A Lightweight End-to-End Model-based Method for Medical Waste Management](https://www.sciencedirect.com/journal/waste-management)  
*(Note: This link is provided for reference and will be updated if necessary.)*

---

## Prerequisites

MedBinNet is developed based on the [MMDetection framework](https://mmdetection.readthedocs.io/en/latest/get_started.html). Before running the model, please ensure that your environment is set up according to the MMDetection installation guide.

- **MMDetection**: Follow the [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) to install all necessary dependencies.

---

## Data Preparation

The Medbin dataset is available for download from [Roboflow](https://universe.roboflow.com/uob-ylti8/medbin_dataset). After downloading, unzip the data into the following directory:

```bash
./MMdetection/data/MedBin_Dataset/
```

Ensure that the dataset structure matches the requirements specified in the MMDetection framework.

---

## Training

To train the MedBinNet model, execute the following command:

```bash
python MMDetection/tools/train.py ./configs/MedBinNet.py
```

This command launches the training process using the provided configuration file. The training script will automatically utilize your prepared dataset and configured model parameters.

---

## Testing

Once training is complete, you can evaluate the model using the testing script:
We provide a trained MedBinNet checkpoint that could be used for reproducing our results.

```bash
python MMDetection/tools/test.py ./configs/MedBinNet.py ./checkpoints/MedBinNet_checkpoint.pth
```

Replace `./checkpoints/MedBinNet_checkpoint.pth` with the path to your trained model checkpoint if it differs.

---

## Configurations

In addition to the main configuration file `MedBinNet.py`, we provide a collection of configuration files for various models. These can be found under the `./configs` directory. Researchers are encouraged to explore and modify these configurations based on their experimental needs.

---

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{your2025medbin,
  title={MedBin: A Lightweight End-to-End Model-based Method for Medical Waste Management},
  author={Your, Name and Collaborator, Name},
  journal={Waste Management},
  year={2025},
  publisher={Elsevier}
}
```

*(Please update the citation details as needed.)*

---

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE). See the LICENSE file for details.

---

For further questions or feedback, please contact the corresponding author or open an issue on this repository.

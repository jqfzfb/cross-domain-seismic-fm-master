Cross-domain transfer learning framework that adapts vision foundation models for seismic interpretation.
This repository provides the implementation of our seismic-to-vision bridging strategy, efficient adaptation using LoRA and prefix tuning, geological prompting for structural consistency, and task-adaptive decoders for seismic facies segmentation and related interpretation tasks.

🌍 Overview

Modern deep learning models for seismic interpretation typically rely on task-specific architectures and large labeled datasets.
This project introduces a cross-domain transfer learning framework that repurposes vision foundation models (FMs) for seismic understanding by:

Mapping seismic amplitudes into the latent space of pretrained vision backbones

Adapting FMs efficiently using LoRA and prefix tuning

Embedding geological priors (stratigraphic ordering) into the feature space

Supporting multiple tasks such as facies segmentation, structural interpretation, and attribute prediction

✨ Key Features

🪄 Seismic-to-Vision Bridge: Lightweight module converting seismic amplitudes into vision FM embeddings

🔧 Efficient FM Adaptation: LoRA + prefix tuning for low-cost, parameter-efficient learning

🧭 Geological Prompting: Inject stratigraphic constraints into FM latent space

🧩 Task-Adaptive Decoder: Suitable for segmentation or regression tasks

📊 Tested on multiple benchmark datasets with diverse geological settings

📁 Repository Structure
cross-domain-seismic-fm-master/
│
├── models/                 # Bridge, adapters, decoder, LoRA modules
├── sam_backbones/          # Vision foundation model backbones (e.g., SAM)
├── configs/                # Training and data configs
├── scripts/                # Training, evaluation, visualization
├── utils/                  # I/O, transforms, metrics
└── examples/               # Example inference + visualization


（如需我基于你真实目录生成更准确版本，我可以自动解析并写完整体结构。）

📦 Installation
git clone https://github.com/jqfzfb/cross-domain-seismic-fm-master.git
cd cross-domain-seismic-fm-master

pip install -r requirements.txt

📚 Datasets

Example seismic data used in the paper can be downloaded from:

📌 Figshare DOI:
https://doi.org/10.6084/m9.figshare.30702569.v1

Benchmark datasets referenced in the manuscript:

Netherlands F3 block — Zenodo: https://zenodo.org/records/1471548

Parihaka 3D — SEG Wiki: https://wiki.seg.org/wiki/Parihaka-3D

Teapot Dome — SEG open datasets

🔗 Pretrained Foundation Model Weights

SAM (Segment Anything Model) pretrained weights are available from:

https://github.com/facebookresearch/segment-anything

Download the desired checkpoint and place it under:

cross-domain-seismic-fm-master/sam_backbones/

🚀 Quick Start
1. Train
python scripts/train.py --config configs/train_fm.yaml

2. Inference
python scripts/inference.py --input path/to/seismic/section.sgy

3. Visualize
python scripts/visualize.py --run path/to/checkpoints/

🔬 Citation

If you use this repository, please cite the manuscript:

(等你论文的正式 BibTeX，我可以帮你生成完整引用格式)

📝 License

Specify your license here (MIT, Apache-2.0, GPL, etc.)
If你告诉我 preferred license，我可以加进去。

🙌 Acknowledgements

This project builds upon:

SAM (Meta AI)

PyTorch

Various open seismic datasets (F3, Parihaka, Teapot Dome)

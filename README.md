# cross_view_dataset
This repository accompanies our paper:

> **Varying Altitude Dataset**  
> (*Under Review at NeurIPS 2025 data track)
## 📦 Dataset Access

You can access the dataset from the following sources:

- 🔗 [Download from Hugging Face](https://huggingface.co/datasets/letsGoBlind/Varying_Altitude_Dataset)
- 🔗 [Download from Kaggle](https://www.kaggle.com/datasets/zhyw86/varying-altitude-dataset/)

Both versions contain the same content: satellite, aerial, and ground-level imagery packaged in ZIP batches.

## 📁 Repository Structure

```bash
.
├── metadata.jsonld        # Croissant-compliant dataset metadata
├── Batch1.zip             # 10 sites of full multi-scale data
├── Batch2.zip             # 10 sites
├── Batch3.zip             # 10 sites
├── ID0102.zip             # A single-site example (small and easy to examine)
└── README.md              # This file

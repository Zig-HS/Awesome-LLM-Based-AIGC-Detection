# Awesome LLM-Based AIGC Detection

A curated list of research papers on detecting AI-Generated Content (AIGC) using Multimodal Large Language Models (MLLMs).

> 📌 **Contributions are welcome!** If you know of a relevant paper not listed here, please submit a PR or open an issue.

---

## Paper List

| Title                                                        |      Venue       | Year |         Modality         |                       Task                       |       New Dataset        |                       Code (& Dataset)                       | Insight |
| :----------------------------------------------------------- | :--------------: | :--: | :----------------------: | :----------------------------------------------: | :----------------------: | :----------------------------------------------------------: | ------- |
| [Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](https://arxiv.org/abs/2503.14853) |       ICML       | 2025 |       Facial Image       |             Detection, Localization              |            -             |       [GitHub](https://github.com/botianzhe/LVLM-DFD)        | -       |
| [BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM](https://arxiv.org/abs/2507.14632) |      arXiv       | 2025 |      Image / Video       |              Detection, Explanation              |       GenBuster++        |          [GitHub](https://github.com/l8cv/BusterX)           | -       |
| [RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection](https://arxiv.org/abs/2508.04524) |      ACM'MM      | 2025 |          Image           |        Detection, Explanation (thinking)         |            -             |                        Not Available                         | -       |
| [FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models](https://arxiv.org/abs/2410.02761) |       ICLR       | 2025 | (PS/Deepfake/AIGC) Image |       Detection, Explanation, Localization       |         MMTD-Set         |       [GitHub](https://github.com/zhipeixu/FakeShield)       | -       |
| [SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model](https://arxiv.org/abs/2412.04292) |       CVPR       | 2025 |          Image           |       Detection, Localization, Explanation       |         SID-Set          |          [GitHub](https://github.com/hzlsaber/SIDA)          | -       |
| [On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection](https://arxiv.org/abs/2410.23623) |     NeurIPS      | 2024 |      Image / Video       |                    Detection                     |           DVF            |     [GitHub](https://github.com/SparkleXFantasy/MM-Det)      | -       |
| [AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](https://arxiv.org/abs/2507.02664) |       ICCV       | 2025 |          Image           |              Detection, Explanation              |        Holmes-Set        |       [GitHub](https://github.com/wyczzy/AIGI-Holmes)        | -       |
| [ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection](https://arxiv.org/abs/2505.11079) |      ACM'MM      | 2025 |          Audio           |                    Detection                     |            -             |   [GitHub](https://github.com/ucas-hao/qwen_audio_for_add)   | -       |
| [Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation](https://arxiv.org/abs/2503.14905) |     NeurIPS      | 2025 |          Image           |              Detection, Explanation              |         FakeClue         |       [GitHub](https://github.com/opendatalab/FakeVLM)       | -       |
| [VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](https://arxiv.org/abs/2510.02282) |      arXiv       | 2025 |          Video           |              Detection, Explanation              |         VidGuard         |    [GitHub](https://github.com/kyoungjunpark/VidGuard-R1)    | -       |
| [Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](https://arxiv.org/abs/2508.21048) |      arXiv       | 2025 |       Facial Image       |              Detection, Explanation              |      HydraFake-100K      |        [GitHub](https://github.com/EricTan7/Veritas)         | -       |
| [UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization](https://arxiv.org/abs/2510.03161) |      arXiv       | 2025 |          Image           |       Detection, Localization, Explanation       |            -             |                        Not Available                         | -       |
| [MIRAGE: Towards AI-Generated Image Detection in the Wild](https://arxiv.org/abs/2508.13223) |      arXiv       | 2025 |          Image           |              Detection, Explanation              |          MIRAGE          |                        Not Available                         | -       |
| [FakeScope: Large Multimodal Expert Model for Transparent AI-Generated Image Forensics](https://arxiv.org/abs/2503.24267) |      arXiv       | 2025 |          Image           |         Detection, Explanation, Instruct         | FakeChain & FakeInstruct |      [GitHub](https://github.com/Yixuanli423/FakeScope)      | -       |
| [LEGION: Learning to Ground and Explain for Synthetic Image Detection](https://arxiv.org/abs/2503.15264) | ICCV (Highlight) | 2025 |          Image           | Detection, Localization, Explanation, Generation |        SynthScars        | [GitHub](https://github.com/opendatalab/LEGION), [HuggingFace Replicate](https://huggingface.co/fanqiNO1/LEGION-8B-replicate) | -       |
| [ThinkFake: Reasoning in Multimodal Large Language Models for AI-Generated Image Detection](https://arxiv.org/abs/2509.19841) |      arXiv       | 2025 |          Image           |              Detection, Explanation              |            -             |                        Not Available                         | -       |

> **Column Descriptions**:
>
> - **Title**: Clickable link to the paper (preferably official publication or arXiv).
> - **Venue**: Conference/journal or `arXiv` if preprint.
> - **Year**: Publication year.
> - **Modality**: Input type the detector handles (e.g., `Image`, `Video`, `Audio`, `Facial Image`).
> - **Task**: Specific detection tasks (e.g., `Detection`, `Localization`, `Explanation`).
> - **New Dataset**: Newly introduced dataset for AIGC detection, or `-` if none.
> - **Code ( & Dataset)**: Link to official implementation and dataset, marked as `GitHub` if available or `Not Available` if not released.
> - **Insight**: Key novelty or finding of the approach.

---

## How to Contribute

1. Fork this repository.
2. Add your paper entry in **alphabetical order by title** (or chronological order if preferred).
3. Ensure all links are valid and insights are concise and informative.
4. Submit a pull request with a brief description.

Please follow the existing format strictly to maintain readability.

---

## License

This repository is licensed under the [MIT License](LICENSE).

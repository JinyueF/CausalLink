# CausalLink

**CausalLink: An Interactive Evaluation Framework for Causal Reasoning**
This repository accompanies our ACL 2025 paper, **"CausalLink: An Interactive Evaluation Framework for Causal Reasoning."**

---

## 🧠 What It Does

CausalLink evaluates causal reasoning in LLMs through:

* Structured scenarios based on causal graphs.
* Controlled interventions and manipulations.
* Automated benchmarking across a variety of model sources (e.g., Hugging Face, OpenAI, Google).

---

## 🔧 Supported Model Sources

* `huggingface`
* `google`
* `deepseek`
* `vec-inf`

The framework automatically handles each model source via a modular pipeline.

---

## 🚀 Getting Started

### Clone and Set Up

```bash
git clone https://github.com/JinyueF/CausalLink.git
cd CausalLink
```

### Run Experiments

Execute the evaluation pipeline using `main.py`:

```bash
python main.py \
  --data_path path/to/your_dataset.csv \
  --result_path path/to/output_results.csv \
  --source huggingface \
  --model your-model-name \
  --model_path path/to/model/or/api \
  --api_key YOUR_ENV_VAR_NAME \
  --temperature 0.6 \
  --num_rep 1 \
  --prompt_template basic
```

### Arguments

| Argument            | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `--data_path`       | Where to save your dataset configurations                                   |
| `--result_path`     | Where to save experiment results                                        |
| `--source`          | Model source: `huggingface`, `google`, `deepseek`, `vec-inf` |
| `--model`           | Model name or API alias                                      |
| `--model_path`      | Path or base URL for model                                   |
| `--api_key`         | Name of the environment variable holding the API key         |
| `--temperature`     | Sampling temperature (default: 0.6)                          |
| `--num_rep`         | Number of experiment repetitions (default: 1)                |
| `--prompt_template` | Key name of the prompt template                 |

---

## 🛠️ Customization

### 1. Tweak the Prompts

Modify or add custom prompts in `prompting_templates.py`. Make sure to use the predefined key-value structure.

### 2. Create a Custom Scenario

Write a subclass of `CausalWorld` (in `dancing_shape.py`) to define a new causal scenario. Don’t forget to create corresponding prompt entries in `prompting_templates.py`.

### 3. Add New Model Pipelines

To support new model sources, extend `pipeline_handler.py` with your own code. Please ensure compatibility with the existing handler interface.

---

## 📄 License

*Specify your license here, e.g., MIT, Apache-2.0, etc.*

---

## ✏️ Citation

If you use this framework in your work, please cite:

```
@inproceedings{
feng2025causallink,
title={CausalLink: An Interactive Evaluation Framework for Causal Reasoning},
author={Jinyue Feng, Frank Rudzicz},
booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
year={2025},
url={https://openreview.net/forum?id=l85lWowbKI}
}
```

---

## 🤝 Contributions

We welcome pull requests, issues, and discussions. Please follow standard GitHub contribution workflows.

---

## 📬 Contact

For questions or feedback, contact: \[[jinyue@cs.toronto.edu](mailto:jinyue@cs.toronto.edu)]

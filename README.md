# 🚀 VIDABench：A COMPREHENSIVE "DYNAMIC-STATIC" EVALUATION FRAMEWORK FOR MOBILE GUIAGENTS IN COMPLEX SERVICE SCENARIOS
VIDABench is a comprehensive evaluation benchmark designed to assess the capabilities of GUI agents/models across five core dimensions: **Grounding, Perception, Single-step Navigation, Multi-step Navigation, and Dynamic Evaluation.**
## 📊 Evaluation Dimensions
1. **Grounding Evaluation**: Evaluates the model's ability to accurately locate specific objects or elements in a visual scene based on textual references. 
   - *Script:* `evaluate_grounding.py`
2. **Perception Evaluation**: Assesses the visual perception capabilities, including element referencing (identifying functions based on given coordinates) and element understanding (attributes, style, function, state, and spatial location).
   - *Script:* `evaluate_perception.py`
3. **Single-step Navigation**: Tests the execution of atomic actions and short-range navigation given a simple, specific instruction.
   - *Script:* `run_singlestep.py` & `evaluate_singlestep.py`
4. **Multi-step Navigation**: Challenges the agent with long-horizon tasks requiring sequential planning and complex reasoning.
   - *Script:*  `run_multistep.py` & `evaluate_multistep.py`
5. **Dynamic Evaluation**: Measures the model's adaptability and interactive performance in dynamic, changing environments.


## 📁 Dataset Preparation

Please download the VIDABench dataset before running the evaluation.

1. Download the dataset from [Google Drive Link].

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
# 填写你需要安装的包，比如：
pip install -r requirements.txt

### 2. Run Evaluation
To evaluate the Grounding capability, run:
python evaluate_grounding.py
To evaluate the Perception capability, run:
python evaluate_perception.py
To evaluate the Single-step Navigation capability, run:
python run_singlestep.py & python evaluate_singlestep.py
To evaluate the Multi-step Navigation capability, run:
python run_multistep.py & python evaluate_multistep.py
To evaluate the Dynamic Navigation capability, please call lanxun

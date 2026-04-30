import sys
sys.path.insert(0, "src")
from collusionlab.runner.config import ExperimentConfig
from collusionlab.runner.sweep import SweepConfig

cfg = ExperimentConfig.from_yaml("configs/pricing_audit.yaml")
print("pricing_audit llm_judge_enabled:", cfg.oversight.llm_judge_enabled)
print("pricing_audit llm_judge_enforcement:", cfg.oversight.llm_judge_enforcement)

for name in [
    "configs/sweep_concealment.yaml",
    "configs/sweep_phase8_part_b_transition.yaml",
    "configs/sweep_phase8_part_b_transition_transcript_proxy.yaml",
]:
    sc = SweepConfig.from_yaml(name)
    print(f"{name}: mode={sc.mode}")

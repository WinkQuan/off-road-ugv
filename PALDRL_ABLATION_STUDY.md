# PAL-DRL Ablation Study

## Ablation Study

### Experimental Setup
To evaluate the individual contribution of each component in PAL-DRL, we conduct a single-factor ablation study. Specifically, based on the full PAL-DRL framework, we remove one component at a time, including Dropout regularization, the local-visibility APF teacher, the posture-aware reward, and the mild exploration mechanism, while keeping all other settings unchanged. All models are trained for 2000 episodes under the same training configuration and evaluated for 100 episodes in the same test environment. To reduce the influence of randomness, each variant is independently repeated under three random seeds, and the final results are reported as mean ± standard deviation.

### Ablation Results
Table X presents the ablation results of PAL-DRL and its four degraded variants in terms of success rate (SR), collision rate (CR), timeout rate (TR), average step (AS), average trajectory length (ATL), average energy consumption (AEC), average posture stability (APS), and average execution time (AET). Overall, the full PAL-DRL achieves the best or the most balanced performance across these metrics, indicating that each proposed component contributes positively to the final navigation performance.

### Analysis
First, after removing Dropout regularization, the model shows degraded performance in SR, CR, and APS, suggesting that Dropout improves the robustness and generalization ability of the policy network in complex off-road environments. Second, when the local-visibility APF teacher is replaced by a full-obstacle APF teacher, the model performance drops noticeably in SR, CR, and AS. This indicates that the local-visibility teacher provides more appropriate supervisory signals that are better aligned with local perception and decision-making.

Furthermore, removing the posture-aware reward causes the most obvious degradation in APS, while SR and ATL also deteriorate to different extents. This demonstrates that posture-aware reward design plays an important role in maintaining vehicle stability and improving overall navigation quality in off-road terrain. Finally, disabling the mild exploration mechanism reduces SR and increases AS and TR, which suggests that proper exploration helps the agent avoid suboptimal behaviors and improves effective policy search during training.

In summary, the Dropout regularization, local-visibility APF teacher, posture-aware reward, and mild exploration mechanism together constitute the key factors behind the performance improvement of PAL-DRL.

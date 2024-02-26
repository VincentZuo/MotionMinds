

1. **Reward Model Training**:
The first part describes training a reward model \( Q \), which uses a prompt and two responses to estimate the preferences of a human. The training objective is to maximize the difference in the predicted reward between the winning response (\( y_{win} \)) and the losing response (\( y_{lose} \)). The loss function is given as:
   \[
   \text{loss} = -\mathbb{E}_{(x, y_{win}, y_{lose})} \left[ \log \left( \frac{r_p(x, y_{win})}{r_p(x, y_{win}) + r_p(x, y_{lose})} \right) \right]
   \]
   Here, \( r_p \) represents the reward model’s prediction, \( x \) is the prompt, and \( y_{win} \) and \( y_{lose} \) are the winning and losing responses, respectively. The expectation \( \mathbb{E} \) is taken over the distribution of these tuples. The objective is to train the reward model to assign higher values to the winning responses compared to the losing ones.

2. **Policy Training via RL**:
The second part describes training a policy model \( \pi_\theta \) that aims to maximize the expected reward as predicted by the reward model. The loss function is based on a KL-divergence regularized objective:
   \[
   \text{loss} = -\mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} \left[ r_p(x, y) \right] + \beta \text{D}_{\text{KL}}\left( \pi_\theta(y|x) || \pi_{ref}(y|x) \right)
   \]
   In this equation, \( \pi_\theta \) is the policy model being trained, \( \pi_{ref} \) is a reference policy (which could be an earlier iteration of the policy or a policy trained on a different objective), and \( \beta \) is a coefficient that balances the reward maximization and KL-divergence regularization. The KL-divergence term ensures that the policy does not diverge too much from the reference policy, which helps in keeping the policy's behavior under control and prevents it from exploiting the reward model in undesirable ways.

Together, these two steps allow for the development of a policy that is informed by human preferences. The reward model is used to capture what humans consider to be good or bad responses to prompts, and the policy model is trained to generate responses that are likely to be rated highly by the reward model. This approach enables the training of AI models that are more aligned with human values and preferences.

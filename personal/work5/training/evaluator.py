"""MetaWorld policy evaluator."""

import os
import numpy as np
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import metaworld
import metaworld.policies as policies
from personal.work2.mw_common.state_injection import make_env_with_fixed_state


class MetaWorldEvaluator:
    def __init__(self, task_name="pick-place-v3", eval_set_path=None):
        self.task_name = task_name
        self.eval_set_path = eval_set_path
        self.eval_set = None

        if eval_set_path and os.path.exists(eval_set_path):
            import json
            with open(eval_set_path) as f:
                self.eval_set = json.load(f)

    def get_expert_policy(self):
        policy_class_name = f"Sawyer{self.task_name.replace('-', ' ').title().replace(' ', '')}Policy"
        policy_class = getattr(policies, policy_class_name)
        return policy_class()

    def run_evaluation(self, model, processor, n_episodes=100, seed=42):
        """Evaluate policy in MetaWorld, return success rate."""
        if self.eval_set is None:
            return self._random_evaluation(model, processor, n_episodes, seed)
        else:
            return self._fixed_evaluation(model, processor, seed)

    def _random_evaluation(self, model, processor, n_episodes, seed):
        rng = np.random.default_rng(seed)
        success_count = 0

        for i in range(n_episodes):
            s = int(rng.integers(0, 100000))
            mt1 = metaworld.MT1(self.task_name, seed=s)
            env = mt1.train_classes[self.task_name](render_mode="rgb_array", camera_name="corner2")
            env.set_task(mt1.train_tasks[0])
            env._freeze_rand_vec = True

            obs, info = env.reset()
            success = False

            for step in range(500):
                action = self._get_model_action(model, processor, obs, env)
                if action is None:
                    expert = self.get_expert_policy()
                    action = expert.get_action(obs)

                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("success", 0):
                    success = True
                if terminated or truncated:
                    break

            if success:
                success_count += 1
            env.close()

        return success_count / n_episodes * 100

    def _fixed_evaluation(self, model, processor, seed):
        states = self.eval_set["states"]
        success_count = 0

        for i, state in enumerate(states[:100]):
            obj = np.array(state["obj_pos"])
            goal = np.array(state["goal_pos"])
            rand_vec = np.concatenate([obj, goal])

            try:
                env, _, _ = make_env_with_fixed_state(
                    self.task_name, rand_vec, seed=seed, camera_name="corner2")
            except Exception:
                continue

            obs, info = env.reset()
            success = False

            for step in range(500):
                action = self._get_model_action(model, processor, obs, env)
                if action is None:
                    expert = self.get_expert_policy()
                    action = expert.get_action(obs)

                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("success", 0):
                    success = True
                if terminated or truncated:
                    break

            if success:
                success_count += 1
            env.close()

        return success_count / min(len(states), 100) * 100

    def _get_model_action(self, model, processor, obs, env):
        """Get action from the trained model."""
        try:
            image = env.render()
            if image is None:
                return None
            image = np.flip(image, (0, 1))

            from PIL import Image
            img = Image.fromarray(image)
            inputs = processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs.to(model.device if hasattr(model, 'device') else 'cuda'))

            if hasattr(outputs, 'logits'):
                action = outputs.logits[0].cpu().numpy()
                if len(action) == 4:
                    return action.astype(np.float32)

            return None
        except Exception:
            return None
import json
import numpy as np


# =========================
# 固定路径
# =========================

EVAL_SET = (
    "personal/work2/fixed_eval_set.json"
)


DATASETS = {

    "corner": {

        "initial_state":
        "personal/work2/dataset_view/"
        "pick_place_corner/"
        "episode_initial_states.json",


        "methods": {

            "ours":
            "personal/work2/duibi/"
            "ours_112_seed42_corner/"
            "subsets/"
            "dynamicanchor_112_seed42.json",


            "ours_v2":
            "personal/work2/duibi/"
            "ours_v2_112_seed42_corner/"
            "subsets/"
            "dynamicanchor_v2_112_seed42.json",


            "uniform":
            "personal/work2/duibi/"
            "uniform_42_corner/"
            "subsets/"
            "uniform_112_seed42.json",


            "random":
            "personal/work2/duibi/"
            "random_42_corner/"
            "subsets/"
            "random_112_seed42.json",
        }
    },


    "corner3": {

        "initial_state":
        "personal/work2/dataset_view/"
        "pick_place_corner3/"
        "episode_initial_states.json",


        "methods": {

            "ours":
            "personal/work2/duibi/"
            "ours_112_seed42_corner3/"
            "subsets/"
            "dynamicanchor_112_seed42.json",


            "ours_v2":
            "personal/work2/duibi/"
            "ours_v2_112_seed42_corner3/"
            "subsets/"
            "dynamicanchor_v2_112_seed42.json",


            "uniform":
            "personal/work2/duibi/"
            "uniform_42_corner3/"
            "subsets/"
            "uniform_112_seed42.json",


            "random":
            "personal/work2/duibi/"
            "random_42_corner3/"
            "subsets/"
            "random_112_seed42.json",
        }
    }

}



# =========================
# 读取数据
# =========================


def load_eval(path):

    with open(path) as f:
        data = json.load(f)

    return data["states"]



def load_subset_states(
        initial_state_file,
        subset_file):

    # 完整episode状态
    with open(initial_state_file) as f:
        episodes = json.load(f)["episodes"]


    # subset index
    with open(subset_file) as f:
        subset = json.load(f)


    indices = subset[
        "selected_episode_indices"
    ]


    states = []

    for idx in indices:

        states.append(
            episodes[idx]["obj_init_pos"]
        )


    return np.array(states)



# =========================
# coverage计算
# =========================


def compute_coverage(
        eval_states,
        train_states):

    distances = []


    for state in eval_states:

        eval_obj = np.array(
            state["obj_pos"]
        )


        dist = np.linalg.norm(
            train_states - eval_obj,
            axis=1
        )


        distances.append(
            dist.min()
        )


    return np.array(distances)



# =========================
# 主程序
# =========================


def main():


    eval_states = load_eval(
        EVAL_SET
    )


    print("\nFixed evaluation states:",
          len(eval_states))


    for dataset_name, cfg in DATASETS.items():


        print("\n")
        print("=" * 80)
        print(dataset_name)
        print("=" * 80)


        for method, subset_file in cfg["methods"].items():


            train_states = load_subset_states(
                cfg["initial_state"],
                subset_file
            )


            distances = compute_coverage(
                eval_states,
                train_states
            )


            print(
                f"{method:<10}"
                f" samples={len(train_states):<5}"
                f" mean={distances.mean():.6f} "
                f"median={np.median(distances):.6f} "
                f"p90={np.percentile(distances,90):.6f} "
                f"max={distances.max():.6f}"
            )



if __name__ == "__main__":

    main()
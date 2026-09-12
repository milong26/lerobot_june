#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
def load_rand_vecs(path):
    with open(path,"r") as f:
        data=json.load(f)
    rand_vecs={}
    for ep in data["episodes"]:
        idx=int(ep["episode_index"])
        rand_vecs[idx]=np.array(ep["rand_vec"],dtype=np.float32)
    return rand_vecs
def load_selection(path):
    with open(path,"r") as f:
        data=json.load(f)
    return [int(x) for x in data["selected_episode_indices"]]
def extract_object_target_xy(rand_vecs):
    result={}
    for idx,v in rand_vecs.items():
        obj_xy=v[:2]
        target_xy=v[3:5]
        result[idx]=(obj_xy,target_xy)
    return result
def compute_region_id(position,minimum,maximum,bins):
    region=np.floor(
        (position-minimum)/(maximum-minimum+1e-8)*bins
    ).astype(int)
    region=np.clip(region,0,bins-1)
    return region[0]*bins+region[1]
def build_joint_heatmap(episode_ids,obj_target_xy,object_min,object_max,target_min,target_max,bins):
    heatmap=np.zeros((bins*bins,bins*bins))
    for idx in episode_ids:
        obj,target=obj_target_xy[idx]
        obj_region=compute_region_id(
            obj,
            object_min,
            object_max,
            bins
        )
        target_region=compute_region_id(
            target,
            target_min,
            target_max,
            bins
        )
        heatmap[obj_region,target_region]+=1
    return heatmap
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate-pool",required=True)
    parser.add_argument("--selection-grid")
    parser.add_argument("--selection-random",required=True)
    parser.add_argument("--selection-deminf",required=True)
    parser.add_argument("--selection-ours",required=True)
    parser.add_argument(
        "--output-dir",
        default="personal/work2/distribution_analysis/results"
    )
    parser.add_argument("--bins",type=int,default=4)
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(exist_ok=True,parents=True)
    rand_vecs=load_rand_vecs(
        Path(args.candidate_pool)
    )
    obj_target_xy=extract_object_target_xy(
        rand_vecs
    )
    all_obj=np.array(
        [x[0] for x in obj_target_xy.values()]
    )
    all_target=np.array(
        [x[1] for x in obj_target_xy.values()]
    )
    object_min=all_obj.min(axis=0)
    object_max=all_obj.max(axis=0)
    target_min=all_target.min(axis=0)
    target_max=all_target.max(axis=0)
    selections={}
    if args.selection_grid:
        selections["Grid-Uniform"]=load_selection(
            args.selection_grid
        )
    selections["Random"]=load_selection(
        args.selection_random
    )
    selections["DemInf"]=load_selection(
        args.selection_deminf
    )
    selections["Ours"]=load_selection(
        args.selection_ours
    )
    heatmaps={}
    for method,indices in selections.items():
        heatmaps[method]=build_joint_heatmap(
            indices,
            obj_target_xy,
            object_min,
            object_max,
            target_min,
            target_max,
            args.bins
        )
    vmax=max(
        np.max(x)
        for x in heatmaps.values()
    )
    fig,axes=plt.subplots(
        2,
        2,
        figsize=(10,9)
    )
    axes=axes.flatten()
    cmap_map={
        "Grid-Uniform":"Blues",
        "Random":"Reds",
        "DemInf":"Oranges",
        "Ours":"Greens"
    }
    for ax,(method,heatmap) in zip(
        axes,
        heatmaps.items()
    ):
        im=ax.imshow(
            heatmap,
            origin="lower",
            cmap=cmap_map[method],
            vmin=0,
            vmax=vmax,
            interpolation="nearest"
        )
        ax.set_title(
            method,
            fontsize=14
        )
        ax.set_xlabel(
            "Target XY Region"
        )
        ax.set_ylabel(
            "Object XY Region"
        )
        ax.set_xticks(
            range(args.bins*args.bins)
        )
        ax.set_yticks(
            range(args.bins*args.bins)
        )
        ax.tick_params(
            labelsize=6
        )
    for ax in axes[len(heatmaps):]:
        ax.axis("off")
    plt.tight_layout()
    save=output/"two_object_joint_configuration_heatmap.png"
    plt.savefig(
        save,
        dpi=300,
        bbox_inches="tight"
    )
    print(save)
if __name__=="__main__":
    main()
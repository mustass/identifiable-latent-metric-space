import os
import argparse


def retrieve_checkpoints(project_dir, model_identifier='best_model.pickle'):
    """
    Inside project_dir there are folders for each of the model runs. Within those folders there is a checkpoint saved as a pickle file. 
    I want to go through all subfolders of project_dir and check which ones have model_identifier existing. 
    For all success cases, I want to append the path to them to a list []
    """ 
    checkpoints_dir = []
    checkpoints_files= []
    for root, dirs, files in os.walk(project_dir):
        if model_identifier in files:
            checkpoints_dir.append(root)
            checkpoints_files.append(model_identifier)
    return checkpoints_dir, checkpoints_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve checkpoints")
    parser.add_argument("--root", type=str, default="/zhome/ef/8/160495/identifiable-latent-metric-space/model_checkpoints/icml25_celeba_baseline")
    parser.add_argument("--checkpoint_label", type=str, default="best_model.pickle")
    args = parser.parse_args()

    dirs, files = retrieve_checkpoints(args.root, args.checkpoint_label)
    print(f"Gathered {len(dirs)} checkpoints")
    print(f"Directories:\n {dirs}")
    print(f"Files:\n {files}")
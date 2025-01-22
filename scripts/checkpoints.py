import os


def retrieve_checkpoints(project_dir, model_identifier='best_model.pickle'):
    """
    Inside project_dir there are folders for each of the model runs. Within those folders there is a checkpoint saved as a pickle file. 
    I want to go through all subfolders of project_dir and check which ones have model_identifier existing. 
    For all success cases, I want to append the path to them to a list []
    """ 


    
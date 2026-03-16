Daily Fantasy Basketball AI Optimizer

This project is meant to be a one-stop-shop for winning in daily fantasy basketball. It collects all up-to-date stats and relevant information, trains a model to predict player stats. It then predicts stats for a given day, generates the optimal team using based on the predictions and the team constraints, and submits the teams to the tournaments.

There are multiple entry-points for different purposes:
* run_all.py: This is the main entrypoint for training the model. Here we define the parameters (architecture, features, and training), and deploy the training session.
* run_full_pipeline.py: This is meant to run each day. It runs the entire pipeline:
  * collect previous stats (code in data_collectors/stats_loaders.py)
  * train model based on previous stats (code in machine_learning_fantasy_basketball_predictions/run_model.py)
  * collect information about tonight's upcoming games
  * before each game, collect up-to-date info such as injuries, starters etc (code in data_collectors/stats_loaders.py)
  * Run predictions to get the predicted stats (code in RunFullPipeline._get_todays_predictions)
  * Generate optimal team based on the predicted stats (code in find_best_team_linear_programming.py)
  * submit the team to the selected tournaments (code in submit_dfs_team.py)

Model training:
The stats-prediction model is based on Pytorch. We wrote a semi-framework around Pytorch, that allows for easy model definition, manipulation, saving and loading. 
In order to train a model we use params, which are all dataclasses and subclasses of AbstractParams, which can be found in machine_learning/params/abstract_params.py
  * Since params are saved as json files, each param class has to implement to_dict and from_dict, which converts objects that we can't save in a json (enums, for example) to a json-saveable format and back again
  * Each model layer, feature type, dataset, metric, callback etc has its own params class, which can be mapped to the actual class. 

#!/bin/bash
# run_dynamics_uniformity_plots

##################################################
##########Execute from data folder################
##################################################

environments=(cartpole reacher pusher ball_in_cup)
environments=(cartpole pusher ball_in_cup)
methods=(random-actions colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies)

mis_path=~/src/model_init_study
path_to_pred_error=~/src/results/pred_error_results

python $mis_path/scripts/plot_dynamics_uniformity.py --init-methods ${methods[*]} --environments ${environments[*]} --path-to-pred-error $path_to_pred_error
cd ..
cpt=$((cpt+1))
echo "finished plotting dynamics uniformity metrics"

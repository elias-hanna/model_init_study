#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
pred_error_plot_upper_limits=(5 5 10000 10000) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1 1 5 5) # warning needs to be in same order as envs

episodes=(5 10 20)
methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies)


environments=(cartpole reacher pusher)
environments=(ball_in_cup)
pred_error_plot_upper_limits=(5 5 5) # warning needs to be in same order as envs
disagr_plot_upper_limits=(5 5 5) # warning needs to be in same order as envs

episodes=(10)
methods=(random-actions colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies)

mis_path=~/src/model_init_study

## Plot means (only means) over replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd ${env}_results
    echo "Processing following folder"; pwd
    python $mis_path/scripts/plot_mean_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --disagr-plot-upper-lim ${disagr_plot_upper_limits[$cpt]}
    cd ..
    cpt=$((cpt+1))
    echo "finished plotting pred errors for $env\n\n"
done

# ## Plot histogram of actions and observations repartition over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
#     python ../../scripts/plot_hist_over_all_single.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path .
#     cd ..
#     cpt=$((cpt+1))
#     echo "finished plotting histograms for $env"
# done

# ## Plot means (only means) over replications on same plot from pretrained models
# cpt=0

# episodes=(20)
# # sup_args=--no-training

# environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
# methods=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies no-init no-init--perfect-model)
# # methods=(random-actions brownian-motion)

# # environments=(ball_in_cup)
# # methods=(random-actions)

# # selection=(all random disagr)
# selection=(random)

# sup_args=('' '--no-training')
# # sup_args=('--no-training' '')

# for sup_arg in "${sup_args[@]}";do
#     for sel in "${selection[@]}";do
#         cd ${sel}${sup_arg}
#         for env in "${environments[@]}"; do
#             cd ${env}${sup_arg}_daqd_results
#             echo "Processing following folder"; pwd
#             python ../../../scripts/pretrained_launch_exp_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --pretrained-data-path ~/results/model_init_study_daqd_results/${sel}${sup_arg} $sup_arg --transfer-selection $sel
#             cd ..
#             cpt=$((cpt+1))
#             echo "finished plotting pred errors for $env"
#             echo ""
#         done
#     done
# done

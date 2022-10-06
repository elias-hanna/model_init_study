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

# ## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
#     python ../../../scripts/plot_mean_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --disagr-plot-upper-lim ${disagr_plot_upper_limits[$cpt]}
#     cd ..
#     cpt=$((cpt+1))
#     echo "finished plotting pred errors for $env\n\n"
# done


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


### BELOW WITH SUP ARGS (ex sigma study res)
#methods=(colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2)

sup_args=(0.1 0.01 0.001)

# ## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     for sup_arg in "${sup_args[@]}"; do
#         cd ${env}_${sup_arg}_results
#         echo "Processing following folder"; pwd
#         python ../../../scripts/plot_mean_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --disagr-plot-upper-lim ${disagr_plot_upper_limits[$cpt]}
#         cd ..
#     done
#     cpt=$((cpt+1))
#     echo "finished plotting pred errors for $env\n\n"
# done

## Plot histogram of actions and observations repartition over replications on same plot
episodes=(20)
cpt=0
for env in "${environments[@]}"; do
    for sup_arg in "${sup_args[@]}"; do
        cd ${env}_${sup_arg}_results
        echo "Processing following folder"; pwd
        python ../../../scripts/plot_hist_over_all_single.py --init-methods ${methods[*]} --init-episode ${episodes[*]} --environment $env --dump-path .
        cd ..
        cpt=$((cpt+1))
        echo "finished plotting histograms for $env"
    done
done


#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

# environments=(ball_in_cup redundant_arm fetch_pick_and_place ant)
# pred_error_plot_upper_limits=(5 1.25 2.5 20) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1 1.25 0.3 20) # warning needs to be in same order as envs

# environments=(ball_in_cup fetch_pick_and_place redundant_arm)
# pred_error_plot_upper_limits=(5 2.5 1.25) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1 0.3 1.25) # warning needs to be in same order as envs

# environments=(ball_in_cup fetch_pick_and_place)
# pred_error_plot_upper_limits=(5 2.5) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1 0.3) # warning needs to be in same order as envs


# environments=(ball_in_cup redundant_arm redundant_arm_no_walls)
# pred_error_plot_upper_limits=(5 1.25 1.25) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1 1.25 1.25) # warning needs to be in same order as envs

# environments=(redundant_arm)
# pred_error_plot_upper_limits=(1.25) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1.25) # warning needs to be in same order as envs

environments=(redundant_arm_no_walls_no_collision)
pred_error_plot_upper_limits=(1.25) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1.25) # warning needs to be in same order as envs

# environments=(fetch_pick_and_place)
# pred_error_plot_upper_limits=(2.5) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(0.3) # warning needs to be in same order as envs

# environments=(ball_in_cup)
# pred_error_plot_upper_limits=(5) # warning needs to be in same order as envs
# disagr_plot_upper_limits=(1) # warning needs to be in same order as envs

reps=(0 1 2 3 4 5 6 7 8 9)
episodes=(5 10 15 20)
# methods=(random-policies random-actions rarph)
methods=(random-policies rarph random-actions)
# methods=(rarph)

# for env in "${environments[@]}"; do 
#     mkdir ${env}_results
#     cd ${env}_results
#     for rep in "${reps[@]}"; do
# 	    mkdir $rep
# 	    cd $rep
# 	    for method in "${methods[@]}"; do
# 	        for ep in "${episodes[@]}"; do
# 		        python ../../../../scripts/launch_exp.py --init-method $method --init-episodes $ep --environment $env --dump-path .
# 	        done
# 	    done
# 	    cd ..
#     done
#     cd ..	
#  done

#  wait

# ## Plot means over replications

# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
# 	for method in "${methods[@]}"; do
# 	    for ep in "${episodes[@]}"; do
#             python ../../scripts/plot_after_run.py --init-method $method --init-episodes $ep --environment $env --dump-path .
#         done
#     done
#     cd ..
# done

# wait

## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
#     python ../../../scripts/plot_mean_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --disagr-plot-upper-lim ${disagr_plot_upper_limits[$cpt]}
#     cd ..
#     cpt=$((cpt+1))
# done

# ## Plot histogram of actions and observations repartition over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
#     python ../../scripts/plot_hist_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path .
#     cd ..
#     cpt=$((cpt+1))
# done

## Plot histogram of actions and observations repartition over replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd ${env}_results
    echo "Processing following folder"; pwd
    python ../../../scripts/plot_hist_over_all_single.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path .
    cd ..
    cpt=$((cpt+1))
    echo "finished plotting histograms for $env"
done

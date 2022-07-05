#!/bin/bash
# run_experiments

##################################################
##########Execute from data folder################
##################################################

environments=(ball_in_cup redundant_arm_no_walls_limited_angles fastsim_maze fastsim_maze_traps)
pred_error_plot_upper_limits=(5 5 50 50) # warning needs to be in same order as envs
disagr_plot_upper_limits=(1 1 1 1) # warning needs to be in same order as envs

# episodes=(5 10 15 20)
# methods=(random-policies random-actions)
# episodes=(5 10 20)
episodes=(5 10 20)
methods=(brownian-motion levy-flight colored-noise-beta-1 colored-noise-beta-2 random-actions)

# for env in "${environments[@]}"; do
# 	for method in "${methods[@]}"; do
# 	    for ep in "${episodes[@]}"; do
#             	# sbatch --partition=cpu-1heure --export=ALL,env=$env,method=$method,ep=$ep model_init_study.sh
#             	sentence=$(sbatch --partition=cpu-1heure --export=ALL,env=$env,method=$method,ep=$ep model_init_study.sh) # get the output from sbatch
#             	stringarray=($sentence)                            # separate the output in words
#             	jobid=(${stringarray[3]})                          # isolate the job ID
#             	sentence="$(squeue -j $jobid)"            # read job's slurm status
#             	stringarray=($sentence) 
#             	jobstatus=(${stringarray[12]})
# 	        while [ "$jobstatus" = "R" ] || [ "$jobstatus" = "PD" ]; do
#                 	sentence="$(squeue -j $jobid)"            # read job's slurm status
#                 	stringarray=($sentence) 
#                 	jobstatus=(${stringarray[12]}) 
#         	done
# 	    done
#         echo "finished $env with $method"
# 	done
# done

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

# ## Plot means (only means) over replications on same plot
# cpt=0
# for env in "${environments[@]}"; do
#     cd ${env}_results
#     echo "Processing following folder"; pwd
#     python ../../scripts/plot_mean_over_all.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path . --pred-err-plot-upper-lim ${pred_error_plot_upper_limits[$cpt]} --disagr-plot-upper-lim ${disagr_plot_upper_limits[$cpt]}
#     cd ..
#     cpt=$((cpt+1))
#     echo "finished plotting pred errors for $env"
# done

## Plot histogram of actions and observations repartition over replications on same plot
cpt=0
for env in "${environments[@]}"; do
    cd ${env}_results
    echo "Processing following folder"; pwd
    python ../../scripts/plot_hist_over_all_single.py --init-methods ${methods[*]} --init-episodes ${episodes[*]} --environment $env --dump-path .
    cd ..
    cpt=$((cpt+1))
    echo "finished plotting histograms for $env"
done

#!/bin/bash
# run_ant_experiments

#environments=(ant ball_in_cup redundant_arm fetch_pick_and_place)
environments=(ant)
reps=(0 1 2 3 4 5 6 7 8 9)
episodes=(5 10 15 20)
methods=(random-policies random-actions)

# for env in "${environments[@]}"; do 
#     mkdir ${env}_results
#     cd ${env}_results
#     for rep in "${reps[@]}"; do
# 	    mkdir $rep
# 	    cd $rep
# 	    for method in "${methods[@]}"; do
# 	        for ep in "${episodes[@]}"; do
# 		        python ../../../scripts/launch_exp.py --init-method $method --init-episodes $ep --environment $env --dump-path .
# 	        done
# 	    done
# 	    cd ..
#     done
#     cd ..	
# done

# wait

## Plot means over replications

for env in "${environments[@]}"; do
    cd ${env}_results
    echo "Processing following folder"; pwd
	for method in "${methods[@]}"; do
	    for ep in "${episodes[@]}"; do
            python ../../scripts/plot_after_run.py --init-method $method --init-episodes $ep --environment $env --dump-path .
        done
    done
    cd ..
done
    

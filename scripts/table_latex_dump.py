import os

def dump_pred_error_latex_table(mean_pred_errors, std_pred_errors, dump_path,
                                env_name, init_methods, init_episodes,
                                pred_steps):
    ## mean_pred_errors and std_pred_errors have shape
    ## (n_init_methods, n_init_episodes, n_pred_steps)
    n_init_methods = mean_pred_errors.shape[0]
    n_init_episodes = mean_pred_errors.shape[1]
    n_pred_steps = mean_pred_errors.shape[2]
    ## Final table is in this format:
    # | Init episodes | Prediction Horizon | INIT 1  | INIT 2 | ... | INIT N |
    # | ---------------------------------------------------------------------|
    # |               |    pred hor 1      | m +- std|   ...  | ... | .....
    # | INIT BUDG 1   |    pred hor 2      | m +- std ...............
    # |               |    pred hor N      |...............................
    # | --------------|--------------------|
    # |               |                    |
    # | INIT BUDG N   |     .....          |
    # |               |                    |
    # |----------------------------------------------------------------------

    ## Open the file
    filename = f'{env_name}_pred_error_table.txt'
    filepath = os.path.join(dump_path, filename)
    f = open(filepath, "w", encoding='utf-8')

    ## Write the tabular headers
    # Create string with the right number of | c | (depends of n_init_methods)
    c_str = ' '.join(['c |' for _ in range(2+n_init_methods)])
    c_str = '| ' + c_str
    f.write(r'\begin{tabular}{'+ c_str + '}')
    f.write('\n')
    f.write('\t')
    f.write(r'\hline')
    f.write('\n')
    f.write('\t')
    f.write(r'\multicolumn{' + str(2+n_init_methods) + r'}{| c |}{' +
            str(env_name) + r'}\\\hline')
    f.write('\n')
    # Format the init_methods names to fit the tabular
    for (name, idx) in zip(init_methods, range(len(init_methods))):
        if name == 'random-actions': init_methods[idx] = 'RA';
        if name == 'random-policies': init_methods[idx] = 'RP';
        if 'colored' in name:
            init_methods[idx] = 'CNRW_' + [int(s) for s in name.split()
                                           if s.isdigit()]

    init_methods_names_str = ' '.join([f' & ${name}$' for name in init_methods])

    f.write('\t')
    f.write(r'\makecell{Initialization \\ episodes} & \makecell{Prediction ' \
            '\\ horizon}' + init_methods_names_str + r' \\\hline')
    f.write('\n')
    ## For each initialization episode budget
    for (init_episode, ep_idx) in zip(init_episodes, range(n_init_episodes)):
        # write the block header
        f.write('\t')
        f.write(r'\multicolumn{1}{| c |}{\multirow{' + str(n_pred_steps)
                + r'}{*}{' + str(init_episode) + r'}}')
        f.write('\n')
        ## For each prediction horizon
        for (pred_step, step_idx) in zip(pred_steps, range(n_pred_steps)):
            to_write = [f'& {mean_pred_errors[init_idx][ep_idx][step_idx]}' +
                        r' $\pm$ ' +
                        f'{std_pred_errors[init_idx][ep_idx][step_idx]}'
                        for init_idx in range(n_init_methods)]
            to_write = ' '.join(to_write)
            to_write = f'& {pred_step} ' + to_write
            to_write = to_write + r' \\\hline'
            f.write('\t')
            f.write(to_write)
            f.write('\n')

    ## Write the tabular close
    f.write(r'\end{tabular}')
    ## Close the file
    f.close()

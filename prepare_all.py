import os
import argparse
import subprocess as sp
import multiprocessing as mp


def start_process(root_path, out_path, ind, start_index, end_index):
    cmd = 'python prepare_dataset.py '
    cmd += f'{root_path} {root_path}/trialinfo {start_index} {end_index} {out_path}/processed_data_{ind}.tfrecord'
    sp.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', type=str, help='root directory of data')
    parser.add_argument('out_path', type=str, help='save directory of processed data')
    parser.add_argument('--split', type=int, default=50, help='split')
    args = parser.parse_args()

    files = [a for a in os.listdir(args.root_path) if a.count('stim_') > 0]
    file_numbers = list(sorted([int(a[5:-4]) for a in files]))
    instances = int(len(file_numbers) / args.split - .001) + 1

    processes = []
    for i in range(instances):
        s = file_numbers[i * args.split:(i + 1) * args.split]
        start_index = min(s)
        end_index = max(s) + 1
        p_args = (args.root_path, args.out_path, i, start_index, end_index)
        p = mp.Process(target=start_process, args=p_args)
        p.start()
        processes.append(p)
    for i in range(instances):
        processes[i].join()


if __name__ == '__main__':
    main()

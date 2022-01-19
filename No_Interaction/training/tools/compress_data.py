import h5py
import numpy as np
import argparse
import os
import tqdm


def process_cpt(h5file, saveall=False):
    assert os.path.exists(h5file)
    outfile = h5file + '.uint8cpt'
    print('process {} -> {}'.format(h5file, outfile))
    f_src = h5py.File(h5file, 'r')
    f_tar = h5py.File(outfile, 'w')
    for _, traj_k in tqdm.tqdm(enumerate(f_src.keys()), total=len(f_src.keys())):
        traj_v = f_src[traj_k]
        # print(traj_v.keys())
        dones = np.asarray(traj_v['dones'])
        if np.any(dones) or saveall:
            f_tar.create_group(traj_k)
            traj_v.copy('actions', f_tar[traj_k], 'actions')
            traj_v.copy('dones', f_tar[traj_k], 'dones')
            traj_v.copy('rewards', f_tar[traj_k], 'rewards')
            f_tar[traj_k].create_group('obs')
            traj_v['obs'].copy('state', f_tar[traj_k]['obs'], 'state')
            f_tar[traj_k]['obs'].create_group('pointcloud')
            pc_src = traj_v['obs']['pointcloud']
            pc_tar = f_tar[traj_k]['obs']['pointcloud']
            pc_src.copy('xyz', pc_tar, 'xyz')
            pc_src.copy('seg', pc_tar, 'seg')
            rgb = np.asarray(pc_src['rgb'])
            rgbuint8 = np.uint8(rgb * 255)
            pc_tar.create_dataset('rgb', rgbuint8.shape, dtype=rgbuint8.dtype, data=rgbuint8)
    print('{} total {} valid trajectories'.format(outfile, len(f_tar.keys())))
    f_src.close()
    f_tar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--filter', type=str)
    parser.add_argument('--all', default=False, action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.input)

    if os.path.isfile(args.input):
        process_cpt(args.input)
    elif os.path.isdir(args.input):
        filelist = os.listdir(args.input)
        if args.filter is None:
            filelist = list(filter(lambda x: x.endswith('.h5') and 'uint8cpt' not in x, filelist))
        else:
            filelist = list(filter(lambda x: args.filter in x and 'uint8cpt' not in x, filelist))
        filelist = sorted(filelist)
        for file in filelist:
            process_cpt(h5file=os.path.join(args.input, file), saveall=args.all)
    else:
        raise NotImplementedError
    pass


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

r"""Script to automatically generate configuration files for the Multi-Agent Tracking Environment.

This script solves the following optimization problem:

.. math::
    \begin{split}
        \operatorname{minimize} ~ & \max_{\vec{x}} \min_{\vec{c}_i} {\left\| \vec{x} - \vec{c}_i \right\|}_2^2, \\
        \text{subject to}       ~ & -1 \preceq \vec{x} \preceq +1, \\
                                & -1 \preceq \vec{c}_i \preceq +1, i = 1, \dots, n \\
    \end{split}

Requirements for this script:
    - torch
    - numpy
    - matplotlib
    - tqdm
"""

# pylint: skip-file

import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

import mate


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SCALE = mate.TERRAIN_SIZE
BASE_CONFIG_FILE = mate.DEFAULT_CONFIG_FILE

MAX_ITERATIONS = 2000
NUM_MESHES = 100


def generate(
    path,
    num_cameras,
    num_targets,
    num_obstacles,
    num_cargoes_per_target=8,
    obstacle_transmittance=0.1,
    seed=0,
    plot=False,
):
    assert num_cargoes_per_target >= 4, (
        f'The number of cargoes per target must be no less than 4. '
        f'Got num_cargoes_per_target = {num_cargoes_per_target}.'
    )
    obstacle_transmittance = max(0.0, min(obstacle_transmittance, 1.0))

    plot = plot and plt is not None
    path = os.path.abspath(path)
    file_ext = os.path.splitext(path)[1].lower()
    assert file_ext in ('.json', '.yaml', '.yml'), f'Unsupported file extension {file_ext}.'

    print(
        'Generating configuration with {} camera{}, {} target{} and {} obstacle{} to "{}".'.format(
            num_cameras,
            ('s' if num_cameras > 1 else ''),
            num_targets,
            ('s' if num_targets > 1 else ''),
            num_obstacles,
            ('s' if num_obstacles > 1 else ''),
            path,
        )
    )

    mate.seed_everything(seed)

    MESH_NUMPY = np.stack(
        np.meshgrid(
            np.linspace(start=-1.0, stop=+1.0, num=NUM_MESHES + 1, endpoint=True),
            np.linspace(start=-1.0, stop=+1.0, num=NUM_MESHES + 1, endpoint=True),
        ),
        axis=-1,
    ).reshape(-1, 2)
    MESH = torch.FloatTensor(MESH_NUMPY[:, np.newaxis, :])
    if torch.cuda.is_available():
        MESH = MESH.cuda()

    if num_cameras > 0:
        locations = 2.0 * torch.rand((num_cameras, 2), dtype=torch.float64) - 1.0
        if torch.cuda.is_available():
            locations = locations.cuda()

        locations.requires_grad_()

        optimizer = optim.Adam([locations], lr=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.25,
            threshold=1e-2,
            threshold_mode='rel',
            patience=16,
            cooldown=8,
            verbose=True,
        )

        if plot:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        else:
            fig = ax1 = ax2 = None
        contour = scatter = line = None
        circles = []

        max_distances = []
        try:
            with tqdm.trange(MAX_ITERATIONS) as pbar:
                for i in pbar:
                    distances = torch.norm(locations - MESH, dim=-1)

                    distance_to_other = torch.norm(locations - locations.unsqueeze(dim=1), dim=-1)
                    distance_to_other = distance_to_other[distance_to_other != 0]

                    distance_horizontal = torch.minimum(
                        (1 - locations[..., 0]).abs(), (1 + locations[..., 0]).abs()
                    )
                    distance_vertical = torch.minimum(
                        (1 - locations[..., 1]).abs(), (1 + locations[..., 1]).abs()
                    )
                    distance_to_border = torch.minimum(distance_horizontal, distance_vertical)

                    distances_to_nearest, indices_to_nearest = torch.min(distances, dim=-1)
                    max_distance, index = torch.max(distances_to_nearest, dim=0)
                    loss = max_distance
                    regularizer = -(
                        0.001 * distance_to_other.min() + 0.1 * distance_to_border.min()
                    )
                    loss += regularizer

                    max_distance_numpy = max_distance.detach().cpu().numpy()
                    locations_numpy = locations.detach().cpu().numpy()

                    max_point = MESH[index, 0].detach().cpu().numpy()
                    max_center = locations[indices_to_nearest[index]].detach().cpu().numpy()

                    pbar.set_postfix({'radius': f'{SCALE * max_distance_numpy:.5f}'})
                    max_distances.append(SCALE * max_distance_numpy)

                    if plot and i % 5 == 0:
                        if i == 0:
                            ax1.set_title(f'{num_cameras} Camera{"s" if num_cameras > 1 else ""}')
                            ax1.set_aspect('equal', 'box')
                            ax1.set_xlim(left=-1.25, right=1.25)
                            ax1.set_ylim(bottom=-1.25, top=1.25)
                            ax1.set_xticks([-1.0, -0.5, 0.0, +0.5, +1.0])
                            ax1.set_xticklabels(['-1000', '-500', '0', '+500', '+1000'])
                            ax1.set_yticks([-1.0, -0.5, 0.0, +0.5, +1.0])
                            ax1.set_yticklabels(['-1000', '-500', '0', '+500', '+1000'])
                            (line,) = ax1.plot([0, 0], [0, 0], linestyle='--', color='black')
                            for location in locations_numpy:
                                c = plt.Circle(
                                    location, radius=max_distance_numpy, zorder=2, fill=False
                                )
                                circles.append(c)
                                ax1.add_patch(c)

                        if contour is not None:
                            for collection in contour.collections:
                                ax1.collections.remove(collection)
                        if scatter is not None:
                            ax1.collections.remove(scatter)
                        color = (
                            1 + indices_to_nearest.detach().cpu().numpy().astype(np.float64)
                        ) / (num_cameras + 1)
                        contour = ax1.tricontourf(
                            *MESH_NUMPY.T,
                            color,
                            levels=num_cameras + 1,
                            vmin=0.0,
                            vmax=1.0,
                            cmap='hsv',
                            zorder=1,
                        )
                        scatter = ax1.scatter(*locations.detach().cpu().numpy().T, color='black')

                        for c, location in zip(circles, locations_numpy):
                            c.set_center(location)
                            c.set_radius(max_distance_numpy)

                        line.set_data([max_center[0], max_point[0]], [max_center[1], max_point[1]])
                        ax2.clear()
                        ax2.plot(max_distances)
                        ax2.set_xlim(left=0.0)
                        ax2.set_ylim(bottom=0.0, top=1.2 * max_distances[0])
                        ax2.set_title(fr'Radius ($r = {SCALE * max_distance_numpy:.5f}$)')
                        ax2.set_xlabel('iteration')
                        ax2.set_ylabel('radius')

                        plt.pause(0.01)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i % 10 == 0:
                        scheduler.step(loss)
                    locations.data.clip_(min=-1.0, max=1.0)

                    if optimizer.param_groups[0]['lr'] < 1e-5:
                        break
        except KeyboardInterrupt:
            pass
    else:
        max_distance_numpy = 0.0
        locations_numpy = np.zeros([num_cameras, 2])

    max_distance_numpy = SCALE * float(max_distance_numpy)
    locations_numpy = (SCALE * np.asarray(locations_numpy, dtype=np.float64)).tolist()

    with open(BASE_CONFIG_FILE, encoding='UTF-8') as file:
        config = yaml.load(file, yaml.SafeLoader)

    config['name'] = f'MultiAgentTracking({num_cameras}v{num_targets}, {num_obstacles})'
    config['num_cargoes_per_target'] = num_cargoes_per_target

    if num_cameras > 0:
        config['camera']['location_random_range'] = []
        for x, y in locations_numpy:
            config['camera']['location_random_range'].append(
                [x - 0.02 * SCALE, x + 0.02 * SCALE, y - 0.02 * SCALE, y + 0.02 * SCALE]
            )
        config['camera']['max_sight_range'] = 2.0 * max_distance_numpy
        config['camera']['radius'] = min(
            config['camera']['radius'], 0.1 * config['camera']['max_sight_range']
        )
    else:
        del config['camera']

    config['target']['location_random_range'] = [
        [-0.5 * SCALE, +0.5 * SCALE, -0.5 * SCALE, +0.5 * SCALE]
    ] * num_targets
    config['target']['sight_range'] = config['camera']['max_sight_range'] / 2.0

    if num_obstacles > 0:
        config['obstacle']['location_random_range'] = [
            [-SCALE, +SCALE, -SCALE, +SCALE]
        ] * num_obstacles

        radius_random_range_min, radius_random_range_max = config['obstacle']['radius_random_range']
        radius_random_range_max = min(
            max(3.0 * radius_random_range_min, 0.15 * max_distance_numpy), radius_random_range_max
        )
        config['obstacle']['radius_random_range'] = [
            radius_random_range_min,
            radius_random_range_max,
        ]
        config['obstacle']['transmittance'] = obstacle_transmittance
    else:
        del config['obstacle']

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass

    with open(path, mode='w') as file:
        if file_ext == '.json':
            json.dump(config, file, indent=2)
        else:
            yaml.dump(config, file, yaml.SafeDumper, indent=2)

    if fig is not None:
        fig.savefig(path[: -len(file_ext)] + '.png')


def main():
    parser = argparse.ArgumentParser(
        prog='python -m mate.assets.generator',
        description='Script to automatically generate configuration files '
        'for the Multi-Agent Tracking Environment.',
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )
    parser.add_argument(
        '--path',
        type=str,
        metavar='PATH',
        default='config.yaml',
        help='Path to save configuration file. (default: %(default)s)',
    )
    parser.add_argument(
        '--num-cameras',
        type=int,
        metavar='CAMERA',
        default=4,
        help='Number of the cameras in the environment. (default: %(default)d)',
    )
    parser.add_argument(
        '--num-targets',
        type=int,
        metavar='TARGET',
        default=8,
        help='Number of the targets in the environment. (default: %(default)d)',
    )
    parser.add_argument(
        '--num-obstacles',
        type=int,
        metavar='OBSTACLE',
        default=0,
        help='Number of the obstacles in the environment. (default: %(default)d)',
    )
    parser.add_argument(
        '--num-cargoes-per-target',
        type=int,
        metavar='CARGO',
        default=8,
        help='Average number of cargoes (>=4) per target. (default: %(default)d)',
    )
    parser.add_argument(
        '--obstacle-transmittance',
        type=float,
        metavar='FACTOR',
        default=0.1,
        help='Transmittance coefficient of obstacles. (default: %(default).2f)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        metavar='SEED',
        default=0,
        help='Random seed for RNGs. (default: %(default)d)',
    )
    parser.add_argument('--plot', action='store_true', help='Show iteration result plots.')
    args = parser.parse_args()

    if plt is None:
        args.plot = False

    generate(**vars(args))


if __name__ == '__main__':
    main()

log_level = 'INFO'
stack_frame = 1
num_heads = 4
agent = dict(
    type='EnsembleBC',
    batch_size=128,
    policy_cfg=dict(
        type='EnsembleContinuousPolicy',
        policy_head_cfg=dict(
            type='DeterministicHead', noise_std=1e-05, ensemble_num=3),
        loss_type='mse_loss',
        nn_cfg=dict(
            type='FastBucketPointFormer',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='FastPointFormerCoreV0',
                xyz_embed_dim=64,
                rgb_embed_dim=64,
                state_diff_embed_dim=64,
                state_other_embed_dim=64,
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape', 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[256, 256, 256],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
                stack_frame=1),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 256, 256],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=256,
                        num_heads=4,
                        latent_dim=32,
                        dropout=0.1),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[256, 1024, 256],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(
                            type='xavier_init', gain=1, bias=0)),
                    dropout=0.1),
                pooling_cfg=dict(embed_dim=256, num_heads=4, latent_dim=32),
                mlp_cfg=None,
                num_blocks=6),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[256, 256, 'action_shape'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.0003, weight_decay=5e-06),
        lr_scheduler_cfg=dict(type='StepLR', step_size=50000, gamma=0.8)))
eval_cfg = dict(
    type='Evaluation',
    num=300,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=False,
    use_log=False,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        stack_frame=1,
        obs_mode='pointcloud',
        reward_type='dense',
        env_name='MoveBucket-v0'))
train_mfrl_cfg = dict(
    on_policy=False,
    total_steps=900000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=150000,
    n_checkpoint=50000,
    init_replay_buffers='',
    init_replay_with_split=[
        '/export/home/v-yehl/dataset/ManiSkill/compressed_data/RefineMoveBucket/',
        '/export/home/v-yehl/net/RL/ManiSkill/mani_skill/assets/config_files/bucket_models.yml'
    ])
env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=1,
    obs_mode='pointcloud',
    reward_type='dense',
    env_name='MoveBucket-v0')
replay_cfg = dict(type='ReplayMemoryRGBInt', capacity=1000000)
work_dir = './work_dirs/bc_transformer_embed_conv_l2_movebucket/EnsembleBC'

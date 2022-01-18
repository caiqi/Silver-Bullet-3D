log_level = 'INFO'
stack_frame = 1
num_heads = 4
dt_state_dim = 13
init_replay_with_split = [
    './full_mani_skill_data/OpenCabinetDoor/',
    '/export/v-qcaii/darwin/ManiSkill/mani_skill/assets/config_files/cabinet_models_door.yml'
]
resume_from = '/export/v-qcaii/darwin/ManiSkill-Learn/work_dirs/ManiSkillModels/door/model_300000.ckpt'
work_dir = './work_dirs/dt_1125_door_v1/DT/DT'
env_name = 'OpenCabinetDoor-v0'
agent = dict(
    type='DT',
    batch_size=32,
    l1_loss=1.0, l2_loss=10.0,
    policy_cfg=dict(
        type='ContinuousDTPolicy',
        policy_head_cfg=dict(type='DeterministicHead', noise_std=1e-05),
        nn_cfg=dict(
            type='PointFormer',
            stack_frame=1,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointFormerCoreV0',
                xyz_embed_dim=128,
                rgb_embed_dim=128,
                state_diff_embed_dim=128,
                state_other_embed_dim=128,
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
        dt_state_dim=13,
        dt_K=8,
        dt_max_ep_len=200,
        dt_embed_dim=32,
        dt_n_layer=4,
        dt_n_head=8,
        dt_attn_pdrop=0.01,
        disable_reward=True,
        pass_through=False,
        original_weight=0.0,
        shuffle_noise=0.0,
        optim_cfg=dict(
            type='AdamW',
            lr=0.001,
            weight_decay=5e-06,
            paramwise_cfg=dict(
                custom_keys=dict(
                    {'backbone.': dict(lr_mult=0.1, decay_mult=0.0)})))))
eval_cfg = dict(
    type='EvaluationDT',
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
        env_name='OpenCabinetDoor-v0'))
train_mfrl_cfg = dict(
    on_policy=False,
    total_steps=300000,
    warm_steps=0,
    n_steps=0,
    n_updates=100,
    n_eval=300000,
    n_checkpoint=5000,
    init_replay_buffers='',
    init_replay_with_split=[
        './full_mani_skill_data/compressed_data/OpenCabinetDoor/',
        '/export/v-qcaii/darwin/ManiSkill/mani_skill/assets/config_files/cabinet_models_door.yml'
    ])
env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=1,
    obs_mode='pointcloud',
    reward_type='dense',
    env_name='OpenCabinetDoor-v0')
replay_cfg = dict(
    type='ReplayMemoryDT',
    capacity=1000000,
    buffer_keys=['obs', 'actions', 'rewards', 'dones'],
    max_ep_len=200,
    scale=50.0,
    K=8,
    compressed=True,
    mode='delayed')

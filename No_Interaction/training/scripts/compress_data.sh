SOURCE_DIR=ManiSkill/original
OUTPUT_DIR=ManiSkill/compress

python No_Interaction/training/tools/compress_data.py --in_dir $SOURCE_DIR/OpenCabinetDrawer --out_dir $OUTPUT_DIR/OpenCabinetDrawer --all
python No_Interaction/training/tools/compress_data.py --in_dir $SOURCE_DIR/OpenCabinetDoor --out_dir $OUTPUT_DIR/OpenCabinetDoor --all

SOURCE_DIR=ManiSkill/state

python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3051-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3051-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3051-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3071-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3071-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3071-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3050-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3050-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3050-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3016-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3016-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3016-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3032-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3032-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3032-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3001-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3001-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3001-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3021-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3021-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3021-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3024-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3024-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3024-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3003-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3003-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3003-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3025-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3025-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3025-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3073-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3073-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3073-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3070-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3070-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3070-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3010-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3010-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3010-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3045-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3045-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3045-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3020-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3020-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3020-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3022-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3022-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3022-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3005-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3005-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3005-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3076-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3076-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3076-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3038-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3038-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3038-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3030-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3030-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3030-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3013-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3013-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3013-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3008-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3008-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3008-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3063-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3063-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3063-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3031-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3031-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3031-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3047-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3047-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3047-v0.h5
python No_Interaction/training/tools/convert_state_pcd_chair.py --env-name PushChair_3027-v0 --traj-name $SOURCE_DIR/PushChair_state/PushChair_3027-v0.h5 --output-name $OUTPUT_DIR/RefinePushChair/PushChair_3027-v0.h5

python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4055-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4055-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4055-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4032-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4032-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4032-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4044-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4044-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4044-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4019-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4019-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4019-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4056-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4056-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4056-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4012-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4012-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4012-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4052-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4052-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4052-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4035-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4035-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4035-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4031-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4031-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4031-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4003-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4003-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4003-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4020-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4020-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4020-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4006-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4006-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4006-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4021-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4021-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4021-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4025-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4025-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4025-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4051-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4051-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4051-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4009-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4009-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4009-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4023-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4023-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4023-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4016-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4016-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4016-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4045-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4045-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4045-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4043-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4043-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4043-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4024-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4024-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4024-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4000-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4000-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4000-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4017-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4017-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4017-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4008-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4008-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4008-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4011-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4011-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4011-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4010-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4010-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4010-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4022-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4022-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4022-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4018-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4018-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4018-v0.h5
python No_Interaction/training/tools/convert_state_pcd_bucket.py --env-name MoveBucket_4001-v0 --traj-name $SOURCE_DIR/MoveBucket_state/MoveBucket_4001-v0.h5 --output-name $OUTPUT_DIR/RefineMoveBucket/MoveBucket_4001-v0.h5


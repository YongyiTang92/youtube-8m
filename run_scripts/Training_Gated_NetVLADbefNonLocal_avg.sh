export CUDA_VISIBLE_DEVICES='0'
python ../train.py \
--train_data_pattern='/mnt/ceph_cv/aicv_image_data/forestlma/Youtube8M_2018/frame/*.tfrecord' \
--model=NetVLADModelLF \
--video_level_classifier_model=willow_MoeModel \
--train_dir=befnorm-nonlocal-gatednetvladLF-128k-1024-80-0002-300iter-norelu-basic-gatedmoe \
--frame_features=True \
--feature_names="rgb,audio" \
--feature_sizes="1024,128" \
--batch_size=80 \
--base_learning_rate=0.0002 \
--netvlad_cluster_size=128 \
--netvlad_hidden_size=1024 \
--moe_l2=1e-6 \
--iterations=300 \
--learning_rate_decay=0.8 \
--netvlad_relu=False \
--gating=True \
--moe_prob_gating=True \
--nonlocalvlad=True \
--beforeNorm=True \
--avg_netvlad=True \
--max_step=700000 \
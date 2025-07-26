# < iemocap >
# bash main.sh {gpu_number} {ckpt_name} {seed} {dataset_name}
# bash main.sh 7 causalmer_iemocap 0 iemocap

GPU=$1
NAME=$2
SEED=$3
DATASET=$4

TODAY=`date +%y%m%d`
FILENAME=${TODAY}_${NAME}

if [ ${DATASET} == "iemocap" ];then
    # iemocap dataset hyperparameters
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --model=MulT \
    --aligned \
    --dataset=iemocap \
    --attn_dropout=0.2 \
    --embed_dropout=0.25 \
    --out_dropout=0.1 \
    --nlevels=5 \
    --num_heads=10 \
    --batch_size=32 \
    --clip=0.8 \
    --optim=Adam \
    --num_epochs=50 \
    --name=${FILENAME} \
    --seed=${SEED} \
    --mod=tav \
    --lr=2e-3 \
    --m_lr=2e-3 \
    --t_lr=1e-6 \
    --a_lr=2e-3 \
    --v_lr=2e-3 \
    --c_t_lr=2e-3 \
    --c_a_lr=2e-3 \
    --c_v_lr=2e-3 \
    --nde_t=1.0 \
    --nde_a=1.0 \
    --nde_v=1.0 \
    --alpha=0.1 \
    --beta=1.0 \
    --constant=0.0 \
    --fusion_mode=mask
fi
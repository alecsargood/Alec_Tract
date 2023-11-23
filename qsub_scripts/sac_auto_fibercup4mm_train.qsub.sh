#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=10:00:00

#$ -S /bin/bash
#$ -j y
#$ -N sac_auto_train_fibercup4mm 
#$ -wd /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/ellie_TTL/logs
#$ -e /cluster/project2/CU-MONDAI/ellie_TTL/logs 

#$ -l tscratch=20G

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

mkdir -p /scratch0/ethompso/$JOB_ID

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

VALIDATION_SUBJECT_ID=fibercup_4mm
SUBJECT_ID=fibercup_4mm
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${VALIDATION_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -rnv "${DATASET_FOLDER}"/datasets/${VALIDATION_SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/
cp -rnv "${DATASET_FOLDER}"/datasets/${SUBJECT_ID} "${WORK_DATASET_FOLDER}"/datasets/

dataset_file=$WORK_DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
validation_dataset_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/${VALIDATION_SUBJECT_ID}.hdf5
reference_file=$WORK_DATASET_FOLDER/datasets/${VALIDATION_SUBJECT_ID}/masks/${VALIDATION_SUBJECT_ID}_wm.nii.gz

# RL params
max_ep=3000 # Chosen empirically
log_interval=50 # Log at n episodes
lr=0.00005 # Learning rate
gamma=0.75 # Gamma for reward discounting

# Model params
prob=0.0 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=100 # Seed per voxel
theta=30 # Maximum angle for streamline curvature

EXPERIMENT=SAC_Auto_FiberCup4mmTrainExp2_dynamic_stepsize_RewardNorm

ID=$(date +"%F-%H_%M_%S")

#seeds=(1111 2222)

#for rng_seed in "${seeds[@]}"
#do
rng_seed=1111
  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python TrackToLearn/trainers/sac_auto_train.py \
    $DEST_FOLDER \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    "${SUBJECT_ID}" \
    "${validation_dataset_file}" \
    "${VALIDATION_SUBJECT_ID}" \
    "${reference_file}" \
    "${SCORING_DATA}" \
    --dynamic_stepsize \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --lr=${lr} \
    --gamma=${gamma} \
    --rng_seed=${rng_seed} \
    --npv=${npv} \
    --theta=${theta} \
    --max_length=1000 \
    --interface_seeding \
    --use_comet \
    --use_gpu \
    --run_tractometer 

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/

  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/



function finish {
    rm -rf /scratch0/ethompso/$JOB_ID
}

trap finish EXIT ERR INT TERM

#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=48:00:00

#$ -S /bin/bash
#$ -j y
#$ -N ismrm_seed9_bonus5
#$ -wd /cluster/project2/CU-MONDAI/Alec_Tract/TrackToLearn

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/Alec_Tract/logs
#$ -e /cluster/project2/CU-MONDAI/Alec_Tract/logs 

#$ -l tscratch=20G

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/Alec_Tract/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

mkdir -p /scratch0/asargood/$JOB_ID

DATASET_FOLDER=${TRACK_TO_LEARN_DATA}/
WORK_DATASET_FOLDER=${LOCAL_TRACK_TO_LEARN_DATA}/

VALIDATION_SUBJECT_ID=ismrm2015
SUBJECT_ID=ismrm2015
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
max_ep=1500 # Chosen empirically
log_interval=50 # Log at n episodes
lr=0.00005 # Learning rate
gamma=0.75 # Gamma for reward discounting
alpha=0.2

# Model params
prob=0.1 # Noise to add to make a prob output. 0 for deterministic

# Env parameters
npv=100 # Seed per voxel
theta=30 # Maximum angle for streamline curvature

Num_Flows=(0 2 4 8 16 32)

bonus=5
EXPERIMENT=ismrm_seed9_bonus5

ID=$(date +"%F-%H_%M_%S")

rng_seed=9999

for num_flows in "${Num_Flows[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"ISMRM"/"$EXPERIMENT"/"$num_flows"

  python TrackToLearn/trainers/NFsac_auto_train.py \
    $DEST_FOLDER \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    "${SUBJECT_ID}" \
    "${validation_dataset_file}" \
    "${VALIDATION_SUBJECT_ID}" \
    "${reference_file}" \
    "${SCORING_DATA}" \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --lr=${lr} \
    --gamma=${gamma} \
    --alpha=${alpha} \
    --num_flows=${num_flows} \
    --rng_seed=${rng_seed} \
    --npv=${npv} \
    --theta=${theta} \
    --target_bonus_factor=${bonus} \
    --max_length=200 \
    --interface_seeding \
    --use_comet \
    --use_gpu \
    --run_tractometer

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done

function finish {
    rm -rf /scratch0/asargood/$JOB_ID
}

trap finish EXIT ERR INT TERM

#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=10:00:00

#$ -S /bin/bash
#$ -j y
#$ -N TTL_validate
#$ -wd /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/ellie_TTL/logs
#$ -e /cluster/project2/CU-MONDAI/ellie_TTL/logs 

set -e

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

# This should point to your dataset folder
DATASET_FOLDER=/cluster/project2/CU-MONDAI/ellie_TTL

# Should be relatively stable
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data

# Data params
dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

n_actor=10000
npv=60
min_length=20
max_length=200

EXPERIMENT=SAC_Auto_ISMRM2015TrainExp2
ID=2023-11-06-11_27_26
SEED=5555

subjectids=(ismrm2015)
#seeds=(1111 2222 3333 4444 5555)

# for SEED in "${seeds[@]}"
# do
for SUBJECT_ID in "${subjectids[@]}"
do
  
    EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
    SCORING_DATA=${DATASET_FOLDER}/datasets/${SUBJECT_ID}/scoring_data
    DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

    dataset_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
    reference_file=$DATASET_FOLDER/datasets/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

    echo $DEST_FOLDER/model/hyperparameters.json
    python3 ${DATASET_FOLDER}/TrackToLearn/TrackToLearn/runners/ttl_validation.py \
      "$DEST_FOLDER" \
      "$EXPERIMENT" \
      "$ID" \
      "${dataset_file}" \
      "${SUBJECT_ID}" \
      "${reference_file}" \
      $DEST_FOLDER/model \
      $DEST_FOLDER/model/hyperparameters.json \
      --npv="${npv}" \
      --interface_seeding \
      --use_gpu \
      --dynamic_stepsize 

    validation_folder=$DEST_FOLDER/scoring_"${prob}"_"${SUBJECT_ID}"_${npv}

    mkdir -p $validation_folder

    mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $validation_folder/

    python3 ${DATASET_FOLDER}/TrackToLearn/scripts/score_tractogram.py $validation_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
      "$SCORING_DATA" \
      $validation_folder \
      --save_full_vc \
      --save_full_ic \
      --save_full_nc \
      --compute_ic_ib \
      --save_ib \
      --save_vb -f -v
done


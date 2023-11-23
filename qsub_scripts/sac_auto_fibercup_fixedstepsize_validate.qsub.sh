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

#$ -l tscratch=20G

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

mkdir -p /scratch0/ethompso/$JOB_ID

base_dir=/cluster/project2/CU-MONDAI/ellie_TTL

experiment=SAC_Auto_FiberCupTrainExp2_fixed_stepsize_large
id=$(date +"%F-%H_%M_%S")0
dataset=${base_dir}/datasets/fibercup_3mm/fibercup_3mm.hdf5
subject_id=fibercup_3mm
seed_mask=${base_dir}/datasets/fibercup_3mm/maps/interface.nii.gz
policy=${base_dir}/experiments/SAC_Auto_FiberCupTrainExp2_fixed_stepsize_large/2023-11-06-18_20_25/1111/model
hyperparams=${policy}/hyperparameters.json
scoring_data=${base_dir}/datasets/fibercup_3mm/scoring_data

path=${base_dir}/experiments/${experiment}/${id}/validate

mkdir -p ${path}
mkdir -p ${path}/tractometer

python3 ${base_dir}/TrackToLearn/TrackToLearn/runners/ttl_validation.py ${path} ${experiment} ${id} ${dataset} ${subject_id} ${seed_mask} ${policy} \
${hyperparams} --scoring_data ${scoring_data} --npv=300 

${base_dir}/TrackToLearn/scripts/run_tractometer.sh ${path}/tractogram_${experiment}_${id}_${subject_id}.trk ${scoring_data} ${path}/tractometer


function finish {
    rm -rf /scratch0/ethompso/$JOB_ID
}

trap finish EXIT ERR INT TERM

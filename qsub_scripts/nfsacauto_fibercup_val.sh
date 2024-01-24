#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=10:00:00

#$ -S /bin/bash
#$ -j y
#$ -N nf_seed4_bonus10_val
#$ -wd /cluster/project2/CU-MONDAI/Alec_Tract/TrackToLearn

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/Alec_Tract/logs
#$ -e /cluster/project2/CU-MONDAI/Alec_Tract/logs 

#$ -l tscratch=20G

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/Alec_Tract/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

mkdir -p /scratch0/asargood/$JOB_ID

base_dir=/cluster/project2/CU-MONDAI/Alec_Tract
id=$(date +"%F-%H_%M_%S")0
dataset=${base_dir}/datasets/fibercup_3mm/fibercup_3mm.hdf5
subject_id=fibercup_3mm
seed_mask=${base_dir}/datasets/fibercup_3mm/maps/interface.nii.gz
experiment=nf_seed4_bonus10
Num_Flows=(32)
scoring_data=${base_dir}/datasets/fibercup_3mm/scoring_data

for num_flows in "${Num_Flows[@]}"
do

    policy=${base_dir}/experiments/Fibercup/${experiment}/${num_flows}/model
    hyperparams=${policy}/hyperparameters.json

    path=${base_dir}/experiments/Fibercup/${experiment}/${num_flows}/validate

    mkdir -p ${path}
    mkdir -p ${path}/tractometer

    python3 ${base_dir}/TrackToLearn/TrackToLearn/runners/ttl_validation.py ${path} ${experiment} ${num_flows} ${dataset} ${subject_id} ${seed_mask} ${policy} \
    ${hyperparams} --scoring_data ${scoring_data} --npv=300 

    ${base_dir}/TrackToLearn/scripts/run_tractometer.sh ${path}/tractogram_${experiment}_${num_flows}_${subject_id}.trk ${scoring_data} ${path}/tractometer

done
function finish {
    rm -rf /scratch0/asargood/$JOB_ID
}

trap finish EXIT ERR INT TERM

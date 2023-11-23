#$ -l tmem=40G
#$ -l h_vmem=40G
#$ -l h_rt=10:00:00

#$ -S /bin/bash
#$ -j y
#$ -N test_TTL 
#$ -wd /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn

#$ -l gpu=true
#$ -o /cluster/project2/CU-MONDAI/ellie_TTL/logs
#$ -e /cluster/project2/CU-MONDAI/ellie_TTL/logs 

#$ -l tscratch=20G

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

mkdir -p /scratch0/ethompso/$JOB_ID

/cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/scripts/sac_auto_train_exp2_fibercup_ellietest.sh


function finish {
    rm -rf /scratch0/ethompso/$JOB_ID
}

trap finish EXIT ERR INT TERM

#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=1:00:00

#$ -S /bin/bash
#$ -j y
#$ -N make_dataset 
#$ -wd /cluster/project2/CU-MONDAI/ellie_TTL/datasets

#$ -o /cluster/project2/CU-MONDAI/ellie_TTL/logs
#$ -e /cluster/project2/CU-MONDAI/ellie_TTL/logs 

source /share/apps/source_files/python/python-3.8.5.source
source /cluster/project2/CU-MONDAI/ellie_TTL/TrackToLearn/joc_ttl_test/bin/activate 
source /share/apps/source_files/cuda/cuda-11.0.source

base_dir=/cluster/project2/CU-MONDAI/ellie_TTL/

path=${base_dir}/datasets/fibercup_3mm
config_file=${base_dir}/datasets/fibercup_3mm_config.json
output=${path}/fibercup_3mm

echo $output

python3 ${base_dir}/TrackToLearn/TrackToLearn/datasets/create_dataset.py ${path} ${config_file} ${output} --normalize




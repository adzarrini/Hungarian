make
matrix_dir=matrix
data_dir=data
out_temp=out_temp.txt
out_file=${3}           # out.txt default
do_serial=${1}          # 0 for no serial computation 1 for yes do serial computation
maxmin=${2}             # max or min
if [ "$out_file" == '' ]
then
    out_file=out.txt
fi
all_tests=(10 100 500 1000 2000 3000) #4000 5000 7500 10000 12500 15000 17500 20000)
if [ ! -d "./${data_dir}" ]
then
    echo Creating directory ./${data_dir}/
    mkdir ${data_dir}
fi
if [ ! -d "./${matrix_dir}" ]
then 
    echo Creating directory ./${matrix_dir}/
    mkdir ${matrix_dir}
fi
printf "n\t\tgpu_out\t\tgpu_time\t\tcpu_out\t\tcpu_time\n" >> ${data_dir}/${out_temp}
for i in "${all_tests[@]}"
do
    name=test_${i}.txt
#    echo $i >> out_cpu.txt
    if [ ! $(find "./${matrix_dir}" -name "$name") ]
    then
        echo Creating matrix ./${matrix_dir}/$name
        ./creatrix $i 9 1234 ${name}
    fi
    cpu_out=''
    if [ "$do_serial" == 1 ]
    then
        cpu_out=$(./hungarian_serial ./${matrix_dir}/${name} $maxmin 0)
    fi
    gpu_out=$(./hungarian_parallel ./${matrix_dir}/${name} $maxmin 0)
    parse_cpu=($cpu_out)
    parse_gpu=($gpu_out)
    line="${parse_gpu[0]}\t\t${parse_gpu[1]}\t\t${parse_gpu[2]}\t\t${parse_cpu[1]}\t\t${parse_cpu[2]}\n"
    printf $line >> ${data_dir}/${out_temp}
done
column -t ${data_dir}/${out_temp} > ${data_dir}/${out_file}
rm ${data_dir}/${out_temp}

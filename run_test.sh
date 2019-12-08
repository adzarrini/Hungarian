make
matrix_dir=matrix
data_dir=data
out_temp=out_temp.txt
out_file=out.txt
all_tests=(10 100 500 1000) #2000 3000 4000 5000 10000 15000)
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
printf "n\t\tcpu_out\t\tgpu_out\t\tcpu_time\t\tgpu_time\n" >> ${data_dir}/${out_temp}
for i in "${all_tests[@]}"
do
    name=test_${i}.txt
#    echo $i >> out_cpu.txt
    if [ ! $(find "./${matrix_dir}" -name "$name") ]
    then
        echo Creating matrix ./${matrix_dir}/$name
        ./creatrix $i 9 1234 ${name}
    fi
    cpu_out=$(./hungarian_serial ./${matrix_dir}/${name} max 0)
    gpu_out=$(./hungarian_parallel ./${matrix_dir}/${name} max 0)
    parse_cpu=($cpu_out)
    parse_gpu=($gpu_out)
    line="${parse_cpu[0]}\t\t${parse_cpu[1]}\t\t${parse_gpu[1]}\t\t${parse_cpu[2]}\t\t${parse_gpu[2]}\n"
    printf $line >> ${data_dir}/${out_temp}
done
column -t ${data_dir}/${out_temp} > ${data_dir}/${out_file}
rm ${data_dir}/${out_temp}

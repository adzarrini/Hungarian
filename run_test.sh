make
#rm data/out_cpu.txt
printf "n\t\tcpu_out\t\tgpu_out\t\tcpu_time\t\tgpu_time\n" >> data/out_temp.txt
for i in 10 100 500 1000 #2000 3000 4000 5000 10000 15000
do
    name=test_cpu_${i}.txt
#    echo $i >> out_cpu.txt
#    ./creatrix $i 9 1234 ${name}
    cpu_out=$(./hungarian_serial ./matrix/${name} max 0)
    gpu_out=$(./hungarian_parallel ./matrix/${name} max 0)
    parse_cpu=($cpu_out)
    parse_gpu=($gpu_out)
    line="${parse_cpu[0]}\t\t${parse_cpu[1]}\t\t${parse_gpu[1]}\t\t${parse_cpu[2]}\t\t${parse_gpu[2]}\n"
    printf $line >> data/out_temp.txt
done
column -t data/out_temp.txt > data/out.txt
rm data/out_temp.txt

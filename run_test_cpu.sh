make
rm data/out_cpu.txt
for i in 10 100 500 1000 2000 3000 4000 5000 10000 15000
do
    name=test_cpu_${i}.txt
#    echo $i >> out_cpu.txt
    ./creatrix $i 9 1234 ${name}
    ./hungarian_serial ./matrix/${name} max 0 >> data/out_cpu.txt
done

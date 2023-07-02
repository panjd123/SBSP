./ga -d 20 -p 1000 -g 100 -s 2 -m 0.3 --truncrate 0.1 -r 30 --thread 20 -o ../result/ga >> ../result/ga.txt
python main.py -d 20 -n -o "../result/ga"
./ga -d 40 -p 30000 -g 80 -s 2 -m 0.3 --truncrate 0.02 -r 4000 --thread 20 -o ../result/ga >> ../result/ga.txt
python main.py -d 40 -n -o "../result/ga"
./ga -d 80 -p 30000 -g 400 -s 2 -m 0.3 --truncrate 0.02 -r 200 --thread 20 -o ../result/ga >> ../result/ga.txt
python main.py -d 80 -n -o "../result/ga"
./ga -d 160 -p 30000 -g 1000 -s 2 -m 0.3 --truncrate 0.01 -r 80 --thread 20 -o ../result/ga >> ../result/ga.txt
python main.py -d 160 -n -o "../result/ga"
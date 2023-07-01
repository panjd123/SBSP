./ga -d 20 -p 1000 -g 100 -m 0.3 --tsize 50 -r 30 --thread 1
python main.py -d 20 -n -o "../result/ga"
./ga -d 40 -p 10000 -g 100 -m 0.3 --tsize 50 -r 30 --thread 2
python main.py -d 40 -n -o "../result/ga"
./ga -d 80 -p 50000 -g 100 -m 0.3 --tsize 200 -r 100 --thread 3
python main.py -d 80 -n -o "../result/ga"
./ga -d 160 -p 20000 -g 100 -m 0.3 -t 5 --tsize 100 -r 100 --thread 3
python main.py -d 160 -n -o "../result/ga"

if [ ! -d "../result/opt" ]; then
  mkdir -p "../result/opt"
fi

python main.py -d 20 -t 9999 -g 10 -o "../result/opt" -c
python main.py -d 40 -t 3600 -g 100 -o "../result/opt" -c
python main.py -d 80 -t 7200 -g 200 -o "../result/opt" -c
python main.py -d 160 -t 10800 -g 300 -o "../result/opt" -c
if [ ! -d "../result/twoends" ]; then
  mkdir -p "../result/twoends"
fi

python main.py -d 20 -t 9999 -g 10 -e -o "../result/twoends" -c
python main.py -d 40 -t 3600 -g 100 -e -o "../result/twoends" -c
python main.py -d 80 -t 7200 -g 200 -e -o "../result/twoends" -c
python main.py -d 160 -t 10800 -g 300 -e -o "../result/twoends" -c
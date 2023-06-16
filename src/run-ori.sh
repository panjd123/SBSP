if [ ! -d "../result/approx" ]; then
  mkdir -p "../result/approx"
fi
python main.py -d 20 -t 9999 -g 40 --approx_draft -o "../result/approx" -c
python main.py -d 40 -t 3600 -g 1000 --simple_draft -o "../result/approx" -c
python main.py -d 80 -t 7200 -g 2000 --simple_draft -o "../result/approx" -c
python main.py -d 160 -t 10800 -g 3000 --simple_draft -o "../result/approx" -c
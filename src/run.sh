if [ ! -d "../opt-result" ]; then
  mkdir ../opt-result
fi
python main.py -d 20 -t 9999 -g 20 -o ../opt-result -c
python main.py -d 40 -t 3600 -g 20 -o ../opt-result -c
python main.py -d 80 -t 7200 -g 20 -o ../opt-result -c
python main.py -d 160 -t 10800 -g 20 -o ../opt-result -c
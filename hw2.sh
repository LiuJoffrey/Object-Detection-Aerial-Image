#wget https://www.dropbox.com/s/ivgy7zg45t5ljzr/0_151_simple_yolo.pkl.88
#mv 0_151_simple_yolo.pkl.88 ./models/0_151_simple_yolo.pkl.88

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xZi7TnYsogYnc1d25D972pnN99TLwvor' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xZi7TnYsogYnc1d25D972pnN99TLwvor" -O aug_yolo.pkl.46 && rm -rf /tmp/cookies.txt
mv aug_yolo.pkl.46 ./models/aug_yolo.pkl.46
python baseline_predict.py $1 $2

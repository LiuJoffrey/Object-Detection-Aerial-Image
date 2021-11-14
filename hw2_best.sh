wget https://www.dropbox.com/s/socgyrd7g41fdne/A_improve_yolo.pkl.82
mv A_improve_yolo.pkl.82 ./models/A_improve_yolo.pkl.82
python improved_predict.py $1 $2

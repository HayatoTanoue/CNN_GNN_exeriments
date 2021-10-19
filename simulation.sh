
model=$1
name=$2
resize=$3
echo $model $name $resize

python main.py --model_name $model --adam_lr 0.01 --sort shuffle --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --adam_lr 0.001 --sort shuffle --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --adam_lr 0.01 --sort sort --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --adam_lr 0.001 --sort sort --name $name --num_epoch 50 --resize $resize

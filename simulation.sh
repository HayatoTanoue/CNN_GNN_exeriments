
model=$1
resize=$2
lr=$3
echo $model $resize $lr

# python main.py --model_name $model --adam_lr 0.01 --sort shuffle --name $name --num_epoch 50 --resize $resize & \
# python main.py --model_name $model --adam_lr 0.001 --sort shuffle --name $name --num_epoch 50 --resize $resize & \
# python main.py --model_name $model --adam_lr 0.01 --sort sort --name $name --num_epoch 50 --resize $resize & \
# python main.py --model_name $model --adam_lr 0.001 --sort sort --name $name --num_epoch 50 --resize $resize


python main.py --model_name $model --name poisson --adam_lr $lr --sort shuffle --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --name new_poisson --adam_lr $lr --sort shuffle --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --name poisson --adam_lr $lr --sort sort --name $name --num_epoch 50 --resize $resize & \
python main.py --model_name $model --name new_poisson --adam_lr $lr --sort sort --name $name --num_epoch 50 --resize $resize
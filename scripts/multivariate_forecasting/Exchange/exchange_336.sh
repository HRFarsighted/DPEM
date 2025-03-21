export CUDA_VISIBLE_DEVICES=0

model_name=Cross_Mamba
Times_for_experiment=1
pred_len=336
for Times_for_experiment in 1
do
    for e_layers in  2 3 4 1 
    do   
        for d_model in  128 256 512 1024
        do
            for d_state in  32 16 8 4
            do
                for learning_rate in  0.00005 0.0001 0.0002
                do
                    for batch_size in  16 32 64
                        do
                            for d_ff in  128 256 512 1024 64
                            do
                                for top_k in  3 2 4 5 6 1
                                do
                                    python -u run.py \
                                        --Times_for_experiment $Times_for_experiment \
                                        --is_training 1 \
                                        --root_path ./dataset/exchange_rate/ \
                                        --data_path exchange_rate.csv \
                                        --model_id Exchange_96_336 \
                                        --model $model_name \
                                        --data custom \
                                        --features M \
                                        --seq_len 96 \
                                        --pred_len $pred_len \
                                        --e_layers $e_layers \
                                        --enc_in 8 \
                                        --dec_in 8 \
                                        --c_out 8 \
                                        --des 'Exp' \
                                        --d_model $d_model \
                                        --learning_rate $learning_rate \
                                        --train_epochs 10 \
                                        --d_state $d_state \
                                        --d_ff $d_ff \
                                        --batch_size $batch_size \
                                        --top_k $top_k \
                                        --itr 1 >logs/LongForecasting/exchange/12.6/$Times_for_experiment'_'$model_name'_'$pred_len'_'$e_layers'_'$d_model'_'$learning_rate'_'$d_state'_'$d_ff'_'$batch_size'_'$top_k.log
                                        sleep 1
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
CUDA_VISIBLE_DEVICES=3 \
    python main.py \
    --width 1 --nepoch 30 --milestone_1 15 --milestone_2 25 \
    --num_batches_per_test 1000 --source svhn_extra --target mnist \
    --outf output/svhn_mnist --arch lenet

# CUDA_VISIBLE_DEVICES=1 \
#     python main.py --width 1 --nepoch 30 --milestone_1 15 \
#     --milestone_2 25 --num_batches_per_test 1000 --source svhn_extra --target mnist \
#     --rotation --outf output/svhn_mnist_r
# CUDA_VISIBLE_DEVICES=1 python main.py --width 8 --nepoch 30 --milestone_1 15 --milestone_2 25 --num_batches_per_test 1000 --source svhn_extra --target mnist --rotation --flip --outf output/svhn_mnist_rf
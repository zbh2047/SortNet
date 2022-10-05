python main.py --dataset MNIST --model 'SortMLPModel(depth=5,width=5120,scalar=True,dropout=0.75,identity_val=10.0)' --loss 'mixture(lam0=0.02,lam_end=0.0002)' --p-start 8 --p-end 1000 --epochs 0,0,100,1450,1500 --eps-test 0.3 --eps-train 0.45 -b 512 --lr 0.02 --wd 0.02 --gpu 0
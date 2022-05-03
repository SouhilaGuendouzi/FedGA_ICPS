# Fed_ICPS
The MLP and CNN models are produced by:

python main_nn.py

Federated learning with MLP and CNN is produced by:

python main_fed.py

See the arguments in options.py.

For example:

python server.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0

--all_clients for averaging over all client models

NB: for CIFAR-10, num_channels must be 3.

=> https://github.com/vaseline555/Federated-Averaging-PyTorch


pip install tabulate

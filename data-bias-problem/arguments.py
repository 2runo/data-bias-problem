import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--val-batch-size', type=int, default=50)
parser.add_argument('--n-epochs', type=int, default=30)
parser.add_argument('--model-name', type=str, default='facebook/bart-base')
parser.add_argument('--dataset-name', type=str, default='scitldr')
parser.add_argument('--ckpt-path', type=str, default='')
parser.add_argument('--tech2-pass-p', type=float, default=0.5)  # tech2에서 적용 안 하고 그냥 패스할 확률
parser.add_argument('--lr', type=float, default=1e-5)  # tech2에서 적용 안 하고 그냥 패스할 확률
parser.add_argument('--lr-cycle', type=float, default=2)  # learning rate scheduler num cycles
parser.add_argument('-tech1', action='store_true')
parser.add_argument('-tech2', action='store_true')
parser.add_argument('-eda', action='store_true')  # EDA nlp (선행 연구)
parser.add_argument('-test', action='store_true')
parser.add_argument('-each', action='store_true')  # 매 epoch마다 체크포인트 저장할까?
args = parser.parse_args()
print(args)



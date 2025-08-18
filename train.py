from src.trainer import train_model
import argparse
import wandb


def main(args):
    config = {
        'epochs': args.epochs,
        'train_dir': args.train_dir,
        'test_dir': args.test_dir,
        'batch_size': args.batch_size,
        'device': args.device,
        'lr': args.lr,
        'dataset_name': args.dataset_name,
        'test_dataset': args.test_dataset
    }
    
    train_model(config)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--train_dir', type=str, default='/kaggle/input/cbsd68/CBSD68')
    parser.add_argument('--test_dir', type=str, default='/kaggle/input/cbsd68/CBSD68')
    parser.add_argument('--noise_level', type=int, default=25)
    
    parser.add_argument('--dataset_name', type=str, default='Mnist')
    parser.add_argument('--test_dataset', type=str, default='Mnist-Adv')

    arguments = parser.parse_args()
    main(arguments)

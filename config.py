import argparse

def get_args():
    parser = argparse.ArgumentParser(
        prog="JojoSCL",
        description="JojoSCL: Shrinkage Contrastive Learning for single-cell RNA sequence Clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--name", type=str, default="10X_PBMC", help="Dataset name")
    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--select_gene", type=int, default=2000, help="Number of genes to select")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.9, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    parser.add_argument("--m", type=float, default=0.5, help="Momentum coefficient")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter")
    parser.add_argument("--enc_1", type=int, default=512, help="Dimension of first encoder layer")
    parser.add_argument("--enc_2", type=int, default=256, help="Dimension of second encoder layer")
    parser.add_argument("--enc_3", type=int, default=128, help="Dimension of third encoder layer")
    parser.add_argument("--mlp_dim", type=int, default=64, help="Dimension of MLP output")
    parser.add_argument("--cluster_methods", type=str, default="KMeans", help="Clustering method to use")
    
    return parser.parse_args()

args = get_args()

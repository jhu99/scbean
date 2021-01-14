import argparse

def parse():
	parser = argparse.ArgumentParser(description = "This tool is used to analyse single cell data.")
	parser.add_argument('--version', action='version', version='%(prog)s 0.1')
	
	subparsers = parser.add_subparsers(dest='command',help='commands')
	train_parser = subparsers.add_parser('train', help='train contents')
	train_parser.add_argument('--ref_file','-r', required = True, 
					help='The path to the reference file.')
	train_parser.add_argument('--epochs','-e', type=int, default=3000,
					help='The size of each batch.')
	train_parser.add_argument('--patience','-pt', type=int, default=50,
					help='The size of each batch.')
	train_parser.add_argument('--name_ref','-n', default = "", 
					help='The name of the reference.')
	train_parser.add_argument('--batch_size','-b', type=int, default=128,
					help='The size of each batch.')
					
	predict_parser = subparsers.add_parser('predict', help='predict contents')
	predict_parser.add_argument('--query_file','-q', required = True, 
					help='The path to one or multiple query file.')
	predict_parser.add_argument('--groups','-g', default = "", 
					help='the columns containing group information, e.g. cell types and batches.',type=str)
	
	parser.add_argument('--path_to_result','-p', default = "./test_results/", 
					help='The path where the result files should be written.')
	parser.add_argument('--format','-f', default= "h5ad", 
					choices=['h5ad','10x_mtx','10x_h5'],
					help='The format of input files. It has two options, \'h5ad\' or \'10x_mtx\', or \'10x_h5\'.')
	parser.add_argument('--y_size','-y', type=int, default=10,
					help='The size of vector y.')
	parser.add_argument('--scale','-s', type=bool, default=True,
					help='The size of each batch.')
	parser.add_argument('--min_counts','-mc', type=int, default=50,
					help='The size of each batch.')
	parser.add_argument('--min_genes','-mg', type=int, default=200,
					help='The size of each batch.')
	args = parser.parse_args()
	return args



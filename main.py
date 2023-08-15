import algorithm
import speech
import os,sys,argparse
import utils

def main():
    parser = argparse.ArgumentParser(
        description     = 'Speech Tag Identifier', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training_file', default = 'data/brown-training.txt')
    args   = parser.parse_args()
    train_set = utils.load_dataset(args.training_file)
    
    speech.application(train_set)

if __name__ == "__main__":
    main()
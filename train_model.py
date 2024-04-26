from preprocess_data import *





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data into pickle binaries for DataLoader')
    parser.add_argument('json_directory', type=str, help='Root directory with annotation JSONs.')
    parser.add_argument('image_directory', type=str, help='Directory containing images.')
    parser.add_argument('labels_path', type=str, help='Path to csv file with labels')

    args = parser.parse_args()
    
    preprocess_data(args.json_directory, 
                    args.image_directory, 
                    args.labels_path)
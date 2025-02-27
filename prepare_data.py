import os
import glob

def prepare_data_files():
    """
    Prepare the data files for PACS dataset in Colab environment.
    Updates the paths in the _train.txt, _val.txt, and _test.txt files.
    """
    # Base paths
    base_dir = '/content/AFourier-based-Framework-for-Domain-Generalization'
    datalists_dir = os.path.join(base_dir, 'data/datalists')

    # Make sure the datalists directory exists
    os.makedirs(datalists_dir, exist_ok=True)

    # Get all text files in the datalists directory
    txt_files = glob.glob(os.path.join(datalists_dir, '*.txt'))

    if not txt_files:
        print("No text files found in datalists directory.")
        print("Please ensure you've downloaded the PACS dataset and created the appropriate text files.")
        return False

    # Process each file to update paths
    for file_path in txt_files:
        print(f"Processing {file_path}...")

        # Read the file content
        with open(file_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            # Update the path to match Colab environment
            # Replace any existing path with the Colab path
            parts = line.strip().split()
            if len(parts) >= 2:
                # Extract the relative path part (after PACS/kfold/)
                rel_path_start = line.find('PACS/kfold/')
                if rel_path_start >= 0:
                    rel_path = line[rel_path_start:]
                    # Extract everything up to the first space after the path
                    rel_path = rel_path.split(' ')[0]

                    # Create new path in Colab format
                    new_path = f'/content/AFourier-based-Framework-for-Domain-Generalization/data/images/{rel_path}'

                    # Replace the old path with the new one
                    updated_line = line.replace(parts[0], new_path)
                    updated_lines.append(updated_line)
                else:
                    # If we can't find a standard pattern, keep the line as is
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)

    print("All data files prepared for Colab environment.")
    return True

if __name__ == "__main__":
    prepare_data_files()

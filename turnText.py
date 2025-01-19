import os

def reverse_file_lines_in_folder(folder_path):
    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            # Reverse the lines in the file
            with open(file_path, "r") as file:
                lines = file.readlines()
            reversed_lines = lines[::-1]
            with open(file_path, "w") as file:
                file.writelines(reversed_lines)

            print(f"Reversed lines in file: {file_name}")

# Example usage:
folder_name = ("generatedT2I_50-parallelLines")
folder_path = f"./genT2Image/{folder_name}"
reverse_file_lines_in_folder(folder_path)
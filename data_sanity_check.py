import os

folder = r".\train_frames"

correct = 0
lower = 0
higher = 0
missing = 0

for folder_name in os.listdir(folder):
    folder_path = os.path.join(folder, folder_name)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        num_files = len(files)
        if num_files == 32:
            correct += 1
        elif num_files < 32:
            image_files = sorted([file for file in files if file.startswith("frame_")])
            current_count = len(image_files)
            missing_inst = 32 - num_files
            with open("log.txt", "a") as file:
                file.write(str(folder_path) + "\n")
            last_image = image_files[-1]
            for i in range(missing_inst):
                new_index = current_count + i
                new_file_name = f"frame_{str(new_index).zfill(4)}.jpg"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.system(f"copy {os.path.join(folder_path, last_image)} {new_file_path}")

            lower += 1
            missing += missing_inst
        elif num_files > 32:
            higher += 1

print(f"Correct: {correct}, Lower: {lower}, Higher: {higher}, missing = {missing}")
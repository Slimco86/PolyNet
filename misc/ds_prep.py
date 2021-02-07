import os
import json

def update_file(path):
    files = os.listdir(path)
    for file in files:
        new_data = {}
        if file.endswith('.json'):
            with open(os.path.join(path,file),'r') as f:
                old_data = json.load(f)
            mp = max([len(old_data[i]) for i in old_data.keys()])
            for p in range(mp):
                new_data[f'id{p+1}']={}
                for key in old_data.keys():
                    try:
                        #print(f"{p+1},{key}\n data:{old_data[key][p]}")
                        new_data[f'id{p+1}'][key] = old_data[key][p]
                    except IndexError:
                        if key == "gender" or key == "emotion" or key == "race" or key == "skin":
                            new_data[f'id{p+1}'][key] = "unknown"
                        elif key == "age":
                            new_data[f'id{p+1}'][key] = None
                        else:    
                            new_data[f'id{p+1}'][key] = []
            
            os.remove(os.path.join(path,file))
            with open(os.path.join(path,file), 'w') as fp:
                json.dump(new_data, fp)
                print(f"File written :{str(os.path.join(path,file))}")


if __name__ == "__main__":
    update_file('datasets/Train2021/train')

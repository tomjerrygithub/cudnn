import json
json_file = '/data3/objects365/objects365_Tiny_train.json'
json_data = json.loads(open(json_file,"r").read())
json_data

categories_dict = {}
for cat in json_data["categories"]:
    categories_dict[cat["id"]] = cat["name"]
print(len(categories_dict))
categories_dict

images_dict = {}
for cat in json_data["images"]:
    images_dict[cat["id"]] = cat["file_name"]
print(len(images_dict))
images_dict

bbox_list = {}
print("ss" in bbox_list)
for cat in json_data["annotations"]:
    print(cat["image_id"])
    if cat["image_id"] in bbox_list:
        bbox_list[cat["image_id"]].append([cat["id"],cat["category_id"],categories_dict[cat["category_id"]],cat["bbox"],cat["iscrowd"],cat["area"],cat["image_id"],images_dict[cat["image_id"]]])
    else:
        bbox_list[cat["image_id"]] = [[cat["id"],cat["category_id"],categories_dict[cat["category_id"]],cat["bbox"],cat["iscrowd"],cat["area"],cat["image_id"],images_dict[cat["image_id"]]]]
    #bbox_list.append([cat["id"],cat["category_id"],categories_dict[cat["category_id"]],cat["bbox"],cat["iscrowd"],cat["area"],cat["image_id"],images_dict[cat["image_id"]]])
print(len(bbox_list))
bbox_list

%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
with open("obj365_tiny_train.txt","w") as f:
    for ss in bbox_list:
        temp = ""
        temp += "/data3/objects365/train/"+images_dict[ss]
        for s in bbox_list[ss]:
            x = s[3][0]
            y = s[3][1]
            w = s[3][2]
            h = s[3][3]
            xmin = x 
            ymin = y 
            xmax = x + w
            ymax = y + h
            
            temp += " "
            temp += ",".join([str(a) for a in [xmin,ymin,xmax,ymax]])
            temp += ","
            temp += str(s[1]-301)
#         plt.imshow(plt.imread("/data3/objects365/train/"+images_dict[ss]))  
#         plt.show() 
#         print(temp)
        f.write(temp+"\n")
    
# 探索 iscrowd 探索每张图 显示任意一张图的及bbox
ss = 140988
for s in bbox_list[ss]:
    #img = Image.open("/data3/objects365/train/obj365_train_000000255477.jpg") #+bbox_list[ss][0][-1]
    
    img = plt.imread("/data3/objects365/train/"+s[-1]) 

    print(bbox_list[ss])
    plt.imshow(img)  
    plt.show() 
    
    
with open("obj365_tiny_classes.txt","w") as f:
    for k in sorted(categories_dict.keys()):
        print(k,categories_dict[k])
        f.write(categories_dict[k]+"\n")
